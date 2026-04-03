"""Helpers for optional per-joint 2D occlusion masking."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable
import logging

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    logger = logging.getLogger(__name__)

from dannce.engine.skeletons.utils import load_body_profile


def populate_occlusion_visibility(
    datadict: Dict,
    datadict_3d: Dict,
    cameras: Dict,
    sample_ids: Iterable,
    params: Dict,
    com3d_dict: Dict | None = None,
):
    """Attach per-camera, per-joint visibility masks to datadict entries."""
    if not params.get("exclude_occluded_2d", False):
        return

    mode = params.get("exclude_occluded_2d_mode", "capsule_raycast")
    if mode not in {"capsule_raycast", "learned", "body_side_learned"}:
        raise ValueError(
            f"Unsupported exclude_occluded_2d_mode={mode!r}. "
            "Supported modes are 'capsule_raycast', 'learned', and "
            "'body_side_learned'."
        )

    flat_cameras = _flatten_cameras(cameras)
    valid_sample_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id in datadict and sample_id in datadict_3d
    ]
    if len(valid_sample_ids) == 0 or len(flat_cameras) == 0:
        return

    body_profile_name = _resolve_body_profile_name(params)
    body_profile = load_body_profile(body_profile_name)

    example_pose = _as_pose_matrix(datadict_3d[valid_sample_ids[0]])
    n_keypoints = example_pose.shape[1]

    joint_names = list(body_profile["joint_names"])
    if len(joint_names) < n_keypoints:
        joint_names.extend(
            [f"joint_{idx}" for idx in range(len(joint_names), n_keypoints)]
        )
    else:
        joint_names = joint_names[:n_keypoints]
    anchor_joint_indices = _resolve_anchor_joint_indices(joint_names)

    limbs = [
        tuple(int(v) for v in limb)
        for limb in body_profile["limbs"]
        if max(limb) < n_keypoints
    ]

    threshold_px = float(params.get("exclude_occluded_2d_reproj_guard_px", 20.0))
    reproj_only_from_stored3d = bool(
        params.get("exclude_occluded_2d_reproj_only_from_stored3d", False)
    )
    min_support_views = int(params.get("exclude_occluded_2d_min_support_views", 2))
    use_temporal_reference = bool(
        params.get("exclude_occluded_2d_temporal_reference", False)
    )
    temporal_window = int(params.get("exclude_occluded_2d_temporal_window", 3))
    use_temporal_jump_filter = bool(
        params.get("exclude_occluded_2d_temporal_jump_filter", True)
    )
    temporal_jump_window = int(
        params.get("exclude_occluded_2d_temporal_jump_window", 5)
    )
    temporal_jump_thresh_px = float(
        params.get("exclude_occluded_2d_temporal_jump_thresh_px", 75.0)
    )
    use_com_filter = bool(
        params.get("exclude_occluded_2d_com_filter", False)
    )
    com_filter_window = int(
        params.get("exclude_occluded_2d_com_filter_window", 5)
    )
    com_filter_thresh_px = float(
        params.get("exclude_occluded_2d_com_filter_thresh_px", 60.0)
    )
    com_filter_require_two_sided = bool(
        params.get("exclude_occluded_2d_com_filter_require_two_sided", True)
    )
    use_bone_prior_rescue = bool(
        params.get("exclude_occluded_2d_bone_prior_rescue", False)
    )
    bone_prior_max_z = float(
        params.get("exclude_occluded_2d_bone_prior_max_z", 3.0)
    )
    bone_prior_min_support_bones = int(
        params.get("exclude_occluded_2d_bone_prior_min_support_bones", 1)
    )
    bone_prior_std_floor = float(
        params.get("exclude_occluded_2d_bone_prior_std_floor", 0.05)
    )
    radius_overrides = params.get("exclude_occluded_2d_joint_radius_overrides_mm") or {}
    lookup_sample_ids = [
        sample_id for sample_id in datadict.keys() if sample_id in datadict_3d
    ]
    temporal_sequences, temporal_positions = _build_temporal_sample_lookup(
        lookup_sample_ids, datadict
    )
    bone_priors_2d = (
        _build_normalized_2d_bone_priors(datadict, lookup_sample_ids, limbs)
        if use_bone_prior_rescue
        else {}
    )
    learned_body_model = (
        _build_learned_body_visibility_model(
            datadict_3d=datadict_3d,
            sample_ids=valid_sample_ids,
            joint_names=joint_names,
            params=params,
        )
        if mode in {"learned", "body_side_learned"}
        else None
    )

    per_camera_stats = defaultdict(lambda: {"finite": 0, "excluded": 0})
    exclusion_stats = {
        "learned_body": 0,
        "raycast": 0,
        "reprojection": 0,
        "reprojection_invalid": 0,
        "stored3d": 0,
        "triangulated": 0,
        "temporal": 0,
        "temporal_jump": 0,
        "com_filter": 0,
        "bone_rescue_temporal_jump": 0,
        "bone_rescue_reprojection": 0,
        "kept_insufficient_evidence": 0,
        "nonfinite_2d": 0,
    }

    for sample_id in valid_sample_ids:
        sample_entry = datadict.get(sample_id)
        if not isinstance(sample_entry, dict):
            continue

        pose3d = _as_pose_matrix(datadict_3d[sample_id])
        primitives = None
        if mode == "capsule_raycast":
            primitives = _build_body_primitives(
                pose3d=pose3d,
                joint_names=joint_names,
                limbs=limbs,
                radius_overrides=radius_overrides,
            )
        learned_body_state = (
            _compute_body_frame_state(
                pose3d=pose3d,
                joint_names=joint_names,
                torso_indices=learned_body_model["torso_indices"],
                lr_pairs=learned_body_model["lr_pairs"],
                scale_hint=learned_body_model["global_scale_mm"],
            )
            if learned_body_model is not None
            else None
        )

        visibility = {}
        for cam_name, cam_2d in sample_entry.get("data", {}).items():
            points_2d = _as_2d_points(cam_2d)
            cam_visibility = np.ones(points_2d.shape[1], dtype=bool)

            cam_params = flat_cameras.get(cam_name)
            cam_center = _camera_center_world(cam_params) if cam_params is not None else None

            for joint_idx in range(points_2d.shape[1]):
                point_2d = points_2d[:, joint_idx]
                if not np.isfinite(point_2d).all():
                    cam_visibility[joint_idx] = False
                    exclusion_stats["nonfinite_2d"] += 1
                    continue

                per_camera_stats[cam_name]["finite"] += 1

                if joint_idx >= pose3d.shape[1]:
                    continue

                if use_temporal_jump_filter and temporal_jump_window > 0:
                    jump_error = _temporal_2d_jump_error(
                        sample_id=sample_id,
                        cam_name=cam_name,
                        joint_idx=joint_idx,
                        point_2d=point_2d,
                        datadict=datadict,
                        temporal_sequences=temporal_sequences,
                        temporal_positions=temporal_positions,
                        temporal_window=temporal_jump_window,
                    )
                    if (
                        jump_error is not None
                        and np.isfinite(jump_error)
                        and jump_error > temporal_jump_thresh_px
                    ):
                        if _joint_matches_bone_priors_2d(
                            points_2d=points_2d,
                            joint_idx=joint_idx,
                            cam_name=cam_name,
                            bone_priors_2d=bone_priors_2d,
                            limbs=limbs,
                            min_support_bones=bone_prior_min_support_bones,
                            max_z=bone_prior_max_z,
                            std_floor=bone_prior_std_floor,
                        ):
                            exclusion_stats["bone_rescue_temporal_jump"] += 1
                            continue
                        cam_visibility[joint_idx] = False
                        per_camera_stats[cam_name]["excluded"] += 1
                        exclusion_stats["temporal_jump"] += 1
                        continue

                if use_com_filter and com_filter_window > 0 and cam_params is not None:
                    com_error = _temporal_anchor_relative_error(
                        sample_id=sample_id,
                        cam_name=cam_name,
                        joint_idx=joint_idx,
                        point_2d=point_2d,
                        datadict=datadict,
                        datadict_3d=datadict_3d,
                        cameras=flat_cameras,
                        com3d_dict=com3d_dict,
                        pose3d=pose3d,
                        joint_names=joint_names,
                        anchor_joint_indices=anchor_joint_indices,
                        temporal_sequences=temporal_sequences,
                        temporal_positions=temporal_positions,
                        temporal_window=com_filter_window,
                        require_two_sided=com_filter_require_two_sided,
                    )
                    if (
                        com_error is not None
                        and np.isfinite(com_error)
                        and com_error > com_filter_thresh_px
                    ):
                        cam_visibility[joint_idx] = False
                        per_camera_stats[cam_name]["excluded"] += 1
                        exclusion_stats["com_filter"] += 1
                        continue

                ref_point_3d = pose3d[:, joint_idx]
                ref_source = "stored3d"
                if not np.isfinite(ref_point_3d).all():
                    ref_point_3d, _ = _triangulate_joint_from_other_views(
                        sample_entry=sample_entry,
                        cameras=flat_cameras,
                        joint_idx=joint_idx,
                        target_cam=cam_name,
                        min_support_views=min_support_views,
                    )
                    if ref_point_3d is not None:
                        ref_source = "triangulated"
                    elif use_temporal_reference and temporal_window > 0:
                        ref_point_3d = _temporal_reference_for_joint(
                            sample_id=sample_id,
                            joint_idx=joint_idx,
                            datadict_3d=datadict_3d,
                            temporal_sequences=temporal_sequences,
                            temporal_positions=temporal_positions,
                            temporal_window=temporal_window,
                        )
                        if ref_point_3d is not None:
                            ref_source = "temporal"

                    if ref_point_3d is None:
                        exclusion_stats["kept_insufficient_evidence"] += 1
                        continue

                if cam_params is None or cam_center is None:
                    continue

                if mode in {"learned", "body_side_learned"}:
                    learned_occluded = _is_joint_occluded_by_learned_body(
                        camera_center=cam_center,
                        cam_params=cam_params,
                        target_point=ref_point_3d,
                        joint_idx=joint_idx,
                        body_model=learned_body_model,
                        body_state=learned_body_state,
                    )
                    if learned_occluded:
                        cam_visibility[joint_idx] = False
                        per_camera_stats[cam_name]["excluded"] += 1
                        exclusion_stats["learned_body"] += 1
                        exclusion_stats[ref_source] += 1
                        continue
                elif _is_joint_occluded(
                    camera_center=cam_center,
                    target_point=ref_point_3d,
                    primitives=primitives,
                    target_joint_idx=joint_idx,
                ):
                    cam_visibility[joint_idx] = False
                    per_camera_stats[cam_name]["excluded"] += 1
                    exclusion_stats["raycast"] += 1
                    exclusion_stats[ref_source] += 1
                    continue

                projected_2d = _project_point_to_camera(ref_point_3d, cam_params)
                if projected_2d is None or not np.isfinite(projected_2d).all():
                    if not reproj_only_from_stored3d or ref_source == "stored3d":
                        cam_visibility[joint_idx] = False
                        per_camera_stats[cam_name]["excluded"] += 1
                        exclusion_stats["reprojection_invalid"] += 1
                        exclusion_stats[ref_source] += 1
                    continue

                reproj_error = float(np.linalg.norm(projected_2d - point_2d))
                if reproj_error > threshold_px:
                    if reproj_only_from_stored3d and ref_source != "stored3d":
                        continue
                    if _joint_matches_bone_priors_2d(
                        points_2d=points_2d,
                        joint_idx=joint_idx,
                        cam_name=cam_name,
                        bone_priors_2d=bone_priors_2d,
                        limbs=limbs,
                        min_support_bones=bone_prior_min_support_bones,
                        max_z=bone_prior_max_z,
                        std_floor=bone_prior_std_floor,
                    ):
                        exclusion_stats["bone_rescue_reprojection"] += 1
                        continue
                    cam_visibility[joint_idx] = False
                    per_camera_stats[cam_name]["excluded"] += 1
                    exclusion_stats["reprojection"] += 1
                    exclusion_stats[ref_source] += 1

            visibility[cam_name] = cam_visibility

        sample_entry["visibility"] = visibility

    total_finite = sum(v["finite"] for v in per_camera_stats.values())
    total_excluded = sum(v["excluded"] for v in per_camera_stats.values())
    if total_finite > 0:
        logger.info(
            "exclude_occluded_2d masked {} / {} finite 2D joint observations "
            "({:.2f}%). LearnedBody={}, Raycast={}, reprojection={}, invalid_proj={}, temporal_jump={}, "
            "com_filter={}, bone_rescue_jump={}, bone_rescue_reproj={}, "
            "kept_without_evidence={}.",
            total_excluded,
            total_finite,
            100.0 * total_excluded / total_finite,
            exclusion_stats["learned_body"],
            exclusion_stats["raycast"],
            exclusion_stats["reprojection"],
            exclusion_stats["reprojection_invalid"],
            exclusion_stats["temporal_jump"],
            exclusion_stats["com_filter"],
            exclusion_stats["bone_rescue_temporal_jump"],
            exclusion_stats["bone_rescue_reprojection"],
            exclusion_stats["kept_insufficient_evidence"],
        )
        for cam_name in sorted(per_camera_stats):
            cam_stats = per_camera_stats[cam_name]
            if cam_stats["finite"] == 0:
                continue
            logger.info(
                "exclude_occluded_2d camera {} masked {} / {} finite joints ({:.2f}%).",
                cam_name,
                cam_stats["excluded"],
                cam_stats["finite"],
                100.0 * cam_stats["excluded"] / cam_stats["finite"],
            )


def _resolve_body_profile_name(params: Dict) -> str:
    loss_cfg = params.get("loss", {})
    bone_cfg = loss_cfg.get("BoneLengthLoss", {}) if isinstance(loss_cfg, dict) else {}
    if isinstance(bone_cfg, dict) and "body_profile" in bone_cfg:
        return bone_cfg["body_profile"]
    return params.get("skeleton", "rat23")


def _normalize_joint_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_torso_joint_indices(joint_names: list[str]) -> list[int]:
    torso_indices = []
    for joint_idx, joint_name in enumerate(joint_names):
        normalized = _normalize_joint_name(joint_name)
        if any(
            token in normalized
            for token in ("neck", "spine", "tailbase", "shoulder", "shd", "hip")
        ) and not any(
            token in normalized
            for token in ("elbow", "knee", "paw", "wrist", "ankle", "foot", "hand", "shin")
        ):
            torso_indices.append(joint_idx)

    if len(torso_indices) > 0:
        return torso_indices
    return _resolve_anchor_joint_indices(joint_names)


def _resolve_lr_joint_pairs(joint_names: list[str]) -> list[tuple[int, int]]:
    normalized_lookup = {
        _normalize_joint_name(joint_name): joint_idx
        for joint_idx, joint_name in enumerate(joint_names)
    }
    pairs = []
    for joint_idx, joint_name in enumerate(joint_names):
        normalized = _normalize_joint_name(joint_name)
        if not normalized.endswith("l"):
            continue
        counterpart = normalized[:-1] + "r"
        counterpart_idx = normalized_lookup.get(counterpart)
        if counterpart_idx is not None:
            pairs.append((joint_idx, counterpart_idx))

    if len(pairs) == 0:
        body_profile_name = "mouse19" if len(joint_names) == 19 else None
        if body_profile_name is not None:
            pairs = [
                (1, 2),
                (7, 10),
                (8, 11),
                (9, 12),
                (13, 16),
                (14, 17),
                (15, 18),
            ]
    return pairs


def _resolve_pair_lookup(lr_pairs: list[tuple[int, int]]) -> dict[int, int]:
    lookup = {}
    for left_idx, right_idx in lr_pairs:
        lookup[left_idx] = right_idx
        lookup[right_idx] = left_idx
    return lookup


def _resolve_body_axis_indices(joint_names: list[str]):
    front_indices = []
    back_indices = []
    for joint_idx, joint_name in enumerate(joint_names):
        normalized = _normalize_joint_name(joint_name)
        if any(token in normalized for token in ("snout", "headf", "neckb", "spinef")):
            front_indices.append(joint_idx)
        if any(token in normalized for token in ("tailbase", "tailmid", "tailend", "spinel", "spinem", "hip")):
            back_indices.append(joint_idx)

    if len(front_indices) == 0:
        front_indices = [
            idx for idx, name in enumerate(joint_names) if "spine" in _normalize_joint_name(name)
        ]
    if len(back_indices) == 0:
        back_indices = _resolve_anchor_joint_indices(joint_names)

    return front_indices, back_indices


def _mean_finite_points(pose3d: np.ndarray, joint_indices: list[int]) -> np.ndarray | None:
    points = []
    for joint_idx in joint_indices:
        if joint_idx >= pose3d.shape[1]:
            continue
        point = pose3d[:, joint_idx]
        if np.isfinite(point).all():
            points.append(point)
    if len(points) == 0:
        return None
    return np.mean(np.stack(points, axis=1), axis=1)


def _compute_body_frame_state(
    pose3d: np.ndarray,
    joint_names: list[str],
    torso_indices: list[int],
    lr_pairs: list[tuple[int, int]],
    scale_hint: float | None = None,
):
    pose3d = _as_pose_matrix(pose3d)
    front_indices, back_indices = _resolve_body_axis_indices(joint_names)

    torso_origin = _mean_finite_points(pose3d, torso_indices)
    front_point = _mean_finite_points(pose3d, front_indices)
    back_point = _mean_finite_points(pose3d, back_indices)

    if torso_origin is None:
        torso_origin = _mean_finite_points(pose3d, list(range(pose3d.shape[1])))
    if torso_origin is None or front_point is None or back_point is None:
        return None

    forward = front_point - back_point
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm <= 1e-8:
        return None
    forward = forward / forward_norm

    lateral_vectors = []
    for left_idx, right_idx in lr_pairs:
        if max(left_idx, right_idx) >= pose3d.shape[1]:
            continue
        left_point = pose3d[:, left_idx]
        right_point = pose3d[:, right_idx]
        if np.isfinite(left_point).all() and np.isfinite(right_point).all():
            lateral_vectors.append(left_point - right_point)

    if len(lateral_vectors) == 0:
        return None

    lateral = np.mean(np.stack(lateral_vectors, axis=1), axis=1)
    lateral = lateral - np.dot(lateral, forward) * forward
    lateral_norm = float(np.linalg.norm(lateral))
    if not np.isfinite(lateral_norm) or lateral_norm <= 1e-8:
        return None
    lateral = lateral / lateral_norm

    up = np.cross(forward, lateral)
    up_norm = float(np.linalg.norm(up))
    if not np.isfinite(up_norm) or up_norm <= 1e-8:
        return None
    up = up / up_norm

    lateral = np.cross(up, forward)
    lateral_norm = float(np.linalg.norm(lateral))
    if not np.isfinite(lateral_norm) or lateral_norm <= 1e-8:
        return None
    lateral = lateral / lateral_norm

    scale = forward_norm
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(scale_hint) if scale_hint is not None else np.nan
    if not np.isfinite(scale) or scale <= 1e-8:
        return None

    axes = np.stack((forward, lateral, up), axis=1)
    relative_pose = pose3d - torso_origin[:, np.newaxis]
    coords_body = axes.T @ relative_pose
    coords_body_norm = coords_body / scale

    return {
        "origin": torso_origin,
        "axes": axes,
        "scale": scale,
        "coords_body_norm": coords_body_norm,
    }


def _build_learned_body_visibility_model(
    datadict_3d: Dict,
    sample_ids: Iterable,
    joint_names: list[str],
    params: Dict,
):
    torso_indices = _resolve_torso_joint_indices(joint_names)
    lr_pairs = _resolve_lr_joint_pairs(joint_names)
    if len(torso_indices) == 0 or len(lr_pairs) == 0:
        return None

    torso_quantile = float(params.get("exclude_occluded_2d_learned_torso_quantile", 0.95))
    torso_margin = float(params.get("exclude_occluded_2d_learned_torso_margin", 1.05))
    torso_points = []
    scales = []
    joint_lateral_values = defaultdict(list)
    valid_frames = 0

    for sample_id in sample_ids:
        pose3d = datadict_3d.get(sample_id)
        if pose3d is None:
            continue
        body_state = _compute_body_frame_state(
            pose3d=pose3d,
            joint_names=joint_names,
            torso_indices=torso_indices,
            lr_pairs=lr_pairs,
        )
        if body_state is None:
            continue

        valid_frames += 1
        scales.append(body_state["scale"])
        coords_body_norm = body_state["coords_body_norm"]
        for joint_idx in torso_indices:
            if joint_idx >= coords_body_norm.shape[1]:
                continue
            point = coords_body_norm[:, joint_idx]
            if np.isfinite(point).all():
                torso_points.append(point)
        for joint_idx in range(min(coords_body_norm.shape[1], len(joint_names))):
            lateral_value = coords_body_norm[1, joint_idx]
            if np.isfinite(lateral_value):
                joint_lateral_values[joint_idx].append(float(lateral_value))

    if len(torso_points) < max(8, len(torso_indices)):
        return None

    torso_points = np.stack(torso_points, axis=1)
    torso_radii_norm = np.quantile(np.abs(torso_points), torso_quantile, axis=1)
    torso_radii_norm = np.maximum(torso_radii_norm * torso_margin, 1e-3)
    global_scale_mm = float(np.median(scales)) if len(scales) > 0 else 1.0
    joint_side_signs = np.zeros(len(joint_names), dtype=np.int8)
    for joint_idx, values in joint_lateral_values.items():
        if len(values) == 0:
            continue
        median_lateral = float(np.median(np.asarray(values, dtype=np.float64)))
        if np.isfinite(median_lateral) and abs(median_lateral) > 1e-3:
            joint_side_signs[joint_idx] = 1 if median_lateral > 0 else -1

    for left_idx, right_idx in lr_pairs:
        if left_idx < len(joint_side_signs):
            joint_side_signs[left_idx] = 1
        if right_idx < len(joint_side_signs):
            joint_side_signs[right_idx] = -1

    logger.info(
        "Learned body visibility teacher from {} samples. Torso radii (normalized)={}.",
        valid_frames,
        np.round(torso_radii_norm, 4),
    )

    return {
        "torso_indices": set(torso_indices),
        "lr_pairs": lr_pairs,
        "pair_lookup": _resolve_pair_lookup(lr_pairs),
        "joint_side_signs": joint_side_signs,
        "torso_radii_norm": torso_radii_norm,
        "torso_forward_radius_norm": float(torso_radii_norm[0]),
        "global_scale_mm": global_scale_mm,
    }


def _transform_point_to_body_norm(point_3d: np.ndarray, body_state) -> np.ndarray | None:
    if body_state is None:
        return None
    point = _as_point3(point_3d)
    if point is None or not np.isfinite(point).all():
        return None
    relative = point - body_state["origin"]
    return (body_state["axes"].T @ relative) / body_state["scale"]


def _camera_optical_axis_body(cam_params, body_state) -> np.ndarray | None:
    if cam_params is None or body_state is None:
        return None
    rotation = np.asarray(cam_params.get("R"), dtype=np.float64)
    if rotation.shape != (3, 3):
        return None
    optical_axis_world = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    optical_axis_body = body_state["axes"].T @ optical_axis_world
    optical_axis_norm = float(np.linalg.norm(optical_axis_body))
    if not np.isfinite(optical_axis_norm) or optical_axis_norm <= 1e-8:
        return None
    return optical_axis_body / optical_axis_norm


def _segment_intersects_axis_aligned_ellipsoid(
    start_point: np.ndarray,
    end_point: np.ndarray,
    radii: np.ndarray,
) -> bool:
    start_scaled = np.asarray(start_point, dtype=np.float64) / radii
    end_scaled = np.asarray(end_point, dtype=np.float64) / radii
    delta = end_scaled - start_scaled

    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(start_scaled, delta))
    c = float(np.dot(start_scaled, start_scaled) - 1.0)
    if a <= 1e-12:
        return False

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False

    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    roots = [
        (-b - sqrt_disc) / (2.0 * a),
        (-b + sqrt_disc) / (2.0 * a),
    ]
    return any(1e-6 < root < (1.0 - 1e-6) for root in roots)


def _is_joint_occluded_by_learned_body(
    camera_center: np.ndarray,
    cam_params,
    target_point: np.ndarray,
    joint_idx: int,
    body_model,
    body_state,
) -> bool:
    if body_model is None or body_state is None:
        return False
    if joint_idx in body_model["torso_indices"]:
        return False

    camera_body = _transform_point_to_body_norm(camera_center, body_state)
    target_body = _transform_point_to_body_norm(target_point, body_state)
    if camera_body is None or target_body is None:
        return False
    if not (np.isfinite(camera_body).all() and np.isfinite(target_body).all()):
        return False

    optical_axis_body = _camera_optical_axis_body(cam_params, body_state)
    if optical_axis_body is None:
        return False

    transverse_view = camera_body - np.dot(camera_body, optical_axis_body) * optical_axis_body
    dominant_axis = int(np.argmax(np.abs(transverse_view)))
    if dominant_axis != 1:
        return False

    camera_lateral = float(transverse_view[1])
    if not np.isfinite(camera_lateral) or abs(camera_lateral) <= 1e-6:
        return False

    target_lateral = float(target_body[1])
    joint_side_signs = body_model.get("joint_side_signs")
    side_sign = 0
    if joint_side_signs is not None and joint_idx < len(joint_side_signs):
        side_sign = int(joint_side_signs[joint_idx])
    if side_sign == 0 and np.isfinite(target_lateral) and abs(target_lateral) > 1e-6:
        side_sign = 1 if target_lateral > 0 else -1
    if side_sign == 0 or side_sign * camera_lateral >= 0:
        return False

    forward_radius = float(body_model.get("torso_forward_radius_norm", np.nan))
    if not np.isfinite(forward_radius) or forward_radius <= 1e-6:
        return False

    target_forward = abs(float(target_body[0]))
    if target_forward <= forward_radius:
        return True

    pair_lookup = body_model.get("pair_lookup", {})
    counterpart_idx = pair_lookup.get(joint_idx)
    if counterpart_idx is None:
        return False
    if counterpart_idx >= body_state["coords_body_norm"].shape[1]:
        return False

    counterpart_body = body_state["coords_body_norm"][:, counterpart_idx]
    if not np.isfinite(counterpart_body).all():
        return False

    pair_mid_forward = abs(float(0.5 * (target_body[0] + counterpart_body[0])))
    return pair_mid_forward <= forward_radius


def _sample_lookup_key(sample_id) -> str:
    return str(sample_id)


def _sample_temporal_group(sample_id) -> str:
    sample_str = str(sample_id)
    prefix, remainder = sample_str.split("_", 1) if "_" in sample_str else ("", sample_str)
    if prefix.isdigit():
        return prefix
    return "__default__"


def _sample_frame_value(sample_id, sample_entry: Dict | None) -> float | None:
    if isinstance(sample_entry, dict):
        frames = sample_entry.get("frames", {})
        for frame in frames.values():
            try:
                frame_value = float(frame)
            except (TypeError, ValueError):
                continue
            if np.isfinite(frame_value):
                return frame_value

    sample_str = str(sample_id)
    local_token = sample_str.split("_", 1)[1] if "_" in sample_str else sample_str
    try:
        frame_value = float(local_token)
    except (TypeError, ValueError):
        return None
    return frame_value if np.isfinite(frame_value) else None


def _resolve_anchor_joint_indices(joint_names: list[str]) -> list[int]:
    preferred_tokens = ("neckb", "spinef", "spinem", "tail(base", "tailbase")
    indices = [
        joint_idx
        for joint_idx, joint_name in enumerate(joint_names)
        if any(token in joint_name.lower() for token in preferred_tokens)
    ]
    if len(indices) > 0:
        return indices

    fallback_tokens = ("neck", "spine", "body", "hip", "shoulder")
    indices = [
        joint_idx
        for joint_idx, joint_name in enumerate(joint_names)
        if any(token in joint_name.lower() for token in fallback_tokens)
    ]
    if len(indices) > 0:
        return indices

    return list(range(len(joint_names)))


def _as_point3(point) -> np.ndarray | None:
    point = np.asarray(point, dtype=np.float64).squeeze()
    if point.ndim == 0 or point.size < 3:
        return None
    if point.ndim == 1:
        point = point[:3]
    elif point.ndim == 2:
        if point.shape[0] == 3:
            point = point[:, 0]
        elif point.shape[1] == 3:
            point = point[0]
        else:
            point = point.reshape(-1)[:3]
    else:
        point = point.reshape(-1)[:3]

    if point.shape != (3,):
        point = point.reshape(-1)[:3]
    if point.shape != (3,):
        return None

    return point


def _sample_anchor_3d(
    sample_id,
    pose3d: np.ndarray,
    com3d_dict: Dict | None,
    anchor_joint_indices: list[int],
) -> np.ndarray | None:
    if com3d_dict is not None and sample_id in com3d_dict:
        com3d = _as_point3(com3d_dict[sample_id])
        if com3d is not None and np.isfinite(com3d).all():
            return com3d

    valid_anchor_points = []
    for joint_idx in anchor_joint_indices:
        if joint_idx >= pose3d.shape[1]:
            continue
        point_3d = pose3d[:, joint_idx]
        if np.isfinite(point_3d).all():
            valid_anchor_points.append(point_3d)

    if len(valid_anchor_points) == 0:
        for joint_idx in range(pose3d.shape[1]):
            point_3d = pose3d[:, joint_idx]
            if np.isfinite(point_3d).all():
                valid_anchor_points.append(point_3d)

    if len(valid_anchor_points) == 0:
        return None

    return np.mean(np.stack(valid_anchor_points, axis=1), axis=1)


def _sample_anchor_2d(
    sample_id,
    cam_name: str,
    pose3d: np.ndarray,
    cameras: Dict,
    com3d_dict: Dict | None,
    anchor_joint_indices: list[int],
) -> np.ndarray | None:
    if cam_name not in cameras:
        return None

    anchor_3d = _sample_anchor_3d(
        sample_id=sample_id,
        pose3d=pose3d,
        com3d_dict=com3d_dict,
        anchor_joint_indices=anchor_joint_indices,
    )
    if anchor_3d is None or not np.isfinite(anchor_3d).all():
        return None

    anchor_2d = _project_point_to_camera(anchor_3d, cameras[cam_name])
    if anchor_2d is None or not np.isfinite(anchor_2d).all():
        return None

    return anchor_2d


def _build_normalized_2d_bone_priors(datadict: Dict, sample_ids: Iterable, limbs):
    priors = defaultdict(lambda: [[] for _ in range(len(limbs))])

    for sample_id in sample_ids:
        sample_entry = datadict.get(sample_id)
        if not isinstance(sample_entry, dict):
            continue

        for cam_name, cam_2d in sample_entry.get("data", {}).items():
            points_2d = _as_2d_points(cam_2d)
            frame_scale = _estimate_2d_frame_scale(points_2d, limbs)
            if frame_scale is None:
                continue

            for limb_idx, (joint_a, joint_b) in enumerate(limbs):
                if max(joint_a, joint_b) >= points_2d.shape[1]:
                    continue

                point_a = points_2d[:, joint_a]
                point_b = points_2d[:, joint_b]
                if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
                    continue

                priors[cam_name][limb_idx].append(
                    float(np.linalg.norm(point_a - point_b) / frame_scale)
                )

    summarized = {}
    for cam_name, limb_values in priors.items():
        means = np.full(len(limb_values), np.nan, dtype=np.float64)
        stds = np.full(len(limb_values), np.nan, dtype=np.float64)
        counts = np.zeros(len(limb_values), dtype=np.int32)

        for limb_idx, values in enumerate(limb_values):
            if len(values) == 0:
                continue

            values_arr = np.asarray(values, dtype=np.float64)
            means[limb_idx] = float(np.mean(values_arr))
            stds[limb_idx] = float(np.std(values_arr))
            counts[limb_idx] = int(len(values_arr))

        summarized[cam_name] = {
            "mean": means,
            "std": stds,
            "count": counts,
        }

    return summarized


def _estimate_2d_frame_scale(points_2d: np.ndarray, limbs) -> float | None:
    lengths = []
    for joint_a, joint_b in limbs:
        if max(joint_a, joint_b) >= points_2d.shape[1]:
            continue

        point_a = points_2d[:, joint_a]
        point_b = points_2d[:, joint_b]
        if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
            continue

        length = float(np.linalg.norm(point_a - point_b))
        if np.isfinite(length) and length > 1e-6:
            lengths.append(length)

    if len(lengths) == 0:
        return None

    frame_scale = float(np.median(lengths))
    if not np.isfinite(frame_scale) or frame_scale <= 1e-6:
        return None

    return frame_scale


def _joint_matches_bone_priors_2d(
    points_2d: np.ndarray,
    joint_idx: int,
    cam_name: str,
    bone_priors_2d: Dict,
    limbs,
    min_support_bones: int,
    max_z: float,
    std_floor: float,
) -> bool:
    if len(bone_priors_2d) == 0 or cam_name not in bone_priors_2d:
        return False

    frame_scale = _estimate_2d_frame_scale(points_2d, limbs)
    if frame_scale is None:
        return False

    priors = bone_priors_2d[cam_name]
    z_scores = []
    for limb_idx, (joint_a, joint_b) in enumerate(limbs):
        if joint_idx not in (joint_a, joint_b):
            continue
        if max(joint_a, joint_b) >= points_2d.shape[1]:
            continue

        point_a = points_2d[:, joint_a]
        point_b = points_2d[:, joint_b]
        if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
            continue

        mean = priors["mean"][limb_idx]
        count = priors["count"][limb_idx]
        if not np.isfinite(mean) or count <= 0:
            continue

        std = priors["std"][limb_idx]
        std = max(float(std) if np.isfinite(std) else 0.0, std_floor)
        normalized_length = float(np.linalg.norm(point_a - point_b) / frame_scale)
        z_scores.append(abs(normalized_length - mean) / std)

    if len(z_scores) < min_support_bones:
        return False

    return float(max(z_scores)) <= max_z


def _build_temporal_sample_lookup(sample_ids: Iterable, datadict: Dict):
    sequences = defaultdict(list)
    positions = {}

    for sample_id in sample_ids:
        sample_entry = datadict.get(sample_id)
        frame_value = _sample_frame_value(sample_id, sample_entry)
        if frame_value is None:
            continue

        group = _sample_temporal_group(sample_id)
        sequences[group].append((frame_value, sample_id))

    for group, entries in sequences.items():
        entries.sort(key=lambda item: (item[0], _sample_lookup_key(item[1])))
        for idx, (frame_value, sample_id) in enumerate(entries):
            positions[_sample_lookup_key(sample_id)] = (group, idx, frame_value)

    return sequences, positions


def _temporal_reference_for_joint(
    sample_id,
    joint_idx: int,
    datadict_3d: Dict,
    temporal_sequences: Dict,
    temporal_positions: Dict,
    temporal_window: int,
) -> np.ndarray | None:
    position_info = temporal_positions.get(_sample_lookup_key(sample_id))
    if position_info is None:
        return None

    group, sample_pos, current_frame = position_info
    sequence = temporal_sequences.get(group, [])
    if len(sequence) == 0:
        return None

    prev_ref = _find_temporal_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        joint_idx=joint_idx,
        datadict_3d=datadict_3d,
        temporal_window=temporal_window,
        step=-1,
    )
    next_ref = _find_temporal_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        joint_idx=joint_idx,
        datadict_3d=datadict_3d,
        temporal_window=temporal_window,
        step=1,
    )

    return _interpolate_temporal_reference(
        current_frame=current_frame,
        prev_ref=prev_ref,
        next_ref=next_ref,
        require_two_sided=False,
    )


def _temporal_2d_jump_error(
    sample_id,
    cam_name: str,
    joint_idx: int,
    point_2d: np.ndarray,
    datadict: Dict,
    temporal_sequences: Dict,
    temporal_positions: Dict,
    temporal_window: int,
) -> float | None:
    temporal_ref = _temporal_2d_reference(
        sample_id=sample_id,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        temporal_sequences=temporal_sequences,
        temporal_positions=temporal_positions,
        temporal_window=temporal_window,
    )
    if temporal_ref is None:
        return None

    return float(np.linalg.norm(np.asarray(point_2d, dtype=np.float64) - temporal_ref))


def _temporal_2d_reference(
    sample_id,
    cam_name: str,
    joint_idx: int,
    datadict: Dict,
    temporal_sequences: Dict,
    temporal_positions: Dict,
    temporal_window: int,
) -> np.ndarray | None:
    position_info = temporal_positions.get(_sample_lookup_key(sample_id))
    if position_info is None:
        return None

    group, sample_pos, current_frame = position_info
    sequence = temporal_sequences.get(group, [])
    if len(sequence) == 0:
        return None

    prev_ref = _find_temporal_2d_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        temporal_window=temporal_window,
        step=-1,
    )
    next_ref = _find_temporal_2d_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        temporal_window=temporal_window,
        step=1,
    )

    return _interpolate_temporal_reference(
        current_frame=current_frame,
        prev_ref=prev_ref,
        next_ref=next_ref,
        require_two_sided=True,
    )


def _temporal_anchor_relative_error(
    sample_id,
    cam_name: str,
    joint_idx: int,
    point_2d: np.ndarray,
    datadict: Dict,
    datadict_3d: Dict,
    cameras: Dict,
    com3d_dict: Dict | None,
    pose3d: np.ndarray,
    joint_names: list[str],
    anchor_joint_indices: list[int],
    temporal_sequences: Dict,
    temporal_positions: Dict,
    temporal_window: int,
    require_two_sided: bool,
) -> float | None:
    del joint_names

    current_anchor_2d = _sample_anchor_2d(
        sample_id=sample_id,
        cam_name=cam_name,
        pose3d=pose3d,
        cameras=cameras,
        com3d_dict=com3d_dict,
        anchor_joint_indices=anchor_joint_indices,
    )
    if current_anchor_2d is None:
        return None

    temporal_rel_ref = _temporal_anchor_relative_reference(
        sample_id=sample_id,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        datadict_3d=datadict_3d,
        cameras=cameras,
        com3d_dict=com3d_dict,
        anchor_joint_indices=anchor_joint_indices,
        temporal_sequences=temporal_sequences,
        temporal_positions=temporal_positions,
        temporal_window=temporal_window,
        require_two_sided=require_two_sided,
    )
    if temporal_rel_ref is None:
        return None

    current_rel = np.asarray(point_2d, dtype=np.float64) - current_anchor_2d
    return float(np.linalg.norm(current_rel - temporal_rel_ref))


def _temporal_anchor_relative_reference(
    sample_id,
    cam_name: str,
    joint_idx: int,
    datadict: Dict,
    datadict_3d: Dict,
    cameras: Dict,
    com3d_dict: Dict | None,
    anchor_joint_indices: list[int],
    temporal_sequences: Dict,
    temporal_positions: Dict,
    temporal_window: int,
    require_two_sided: bool,
) -> np.ndarray | None:
    position_info = temporal_positions.get(_sample_lookup_key(sample_id))
    if position_info is None:
        return None

    group, sample_pos, current_frame = position_info
    sequence = temporal_sequences.get(group, [])
    if len(sequence) == 0:
        return None

    prev_ref = _find_temporal_anchor_relative_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        datadict_3d=datadict_3d,
        cameras=cameras,
        com3d_dict=com3d_dict,
        anchor_joint_indices=anchor_joint_indices,
        temporal_window=temporal_window,
        step=-1,
    )
    next_ref = _find_temporal_anchor_relative_neighbor(
        sequence=sequence,
        start_pos=sample_pos,
        current_frame=current_frame,
        cam_name=cam_name,
        joint_idx=joint_idx,
        datadict=datadict,
        datadict_3d=datadict_3d,
        cameras=cameras,
        com3d_dict=com3d_dict,
        anchor_joint_indices=anchor_joint_indices,
        temporal_window=temporal_window,
        step=1,
    )

    return _interpolate_temporal_reference(
        current_frame=current_frame,
        prev_ref=prev_ref,
        next_ref=next_ref,
        require_two_sided=require_two_sided,
    )


def _interpolate_temporal_reference(
    current_frame: float,
    prev_ref,
    next_ref,
    require_two_sided: bool,
):
    if prev_ref is not None and next_ref is not None:
        prev_frame, prev_point = prev_ref
        next_frame, next_point = next_ref
        if next_frame > prev_frame:
            alpha = (current_frame - prev_frame) / (next_frame - prev_frame)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            return (1.0 - alpha) * prev_point + alpha * next_point

    if require_two_sided:
        return None

    if prev_ref is None and next_ref is None:
        return None

    if prev_ref is None:
        return next_ref[1]
    if next_ref is None:
        return prev_ref[1]

    prev_gap = abs(current_frame - prev_ref[0])
    next_gap = abs(next_ref[0] - current_frame)
    return prev_ref[1] if prev_gap <= next_gap else next_ref[1]


def _find_temporal_neighbor(
    sequence,
    start_pos: int,
    current_frame: float,
    joint_idx: int,
    datadict_3d: Dict,
    temporal_window: int,
    step: int,
):
    idx = start_pos + step
    while 0 <= idx < len(sequence):
        frame_value, neighbor_sample_id = sequence[idx]
        if abs(frame_value - current_frame) > temporal_window:
            break

        pose3d = datadict_3d.get(neighbor_sample_id)
        if pose3d is not None:
            pose3d = _as_pose_matrix(pose3d)
            if joint_idx < pose3d.shape[1]:
                point_3d = pose3d[:, joint_idx]
                if np.isfinite(point_3d).all():
                    return frame_value, point_3d

        idx += step

    return None


def _find_temporal_2d_neighbor(
    sequence,
    start_pos: int,
    current_frame: float,
    cam_name: str,
    joint_idx: int,
    datadict: Dict,
    temporal_window: int,
    step: int,
):
    idx = start_pos + step
    while 0 <= idx < len(sequence):
        frame_value, neighbor_sample_id = sequence[idx]
        if abs(frame_value - current_frame) > temporal_window:
            break

        sample_entry = datadict.get(neighbor_sample_id)
        if isinstance(sample_entry, dict):
            cam_points = sample_entry.get("data", {}).get(cam_name)
            if cam_points is not None:
                cam_points = _as_2d_points(cam_points)
                if joint_idx < cam_points.shape[1]:
                    point_2d = cam_points[:, joint_idx]
                    if np.isfinite(point_2d).all():
                        return frame_value, point_2d

        idx += step

    return None


def _find_temporal_anchor_relative_neighbor(
    sequence,
    start_pos: int,
    current_frame: float,
    cam_name: str,
    joint_idx: int,
    datadict: Dict,
    datadict_3d: Dict,
    cameras: Dict,
    com3d_dict: Dict | None,
    anchor_joint_indices: list[int],
    temporal_window: int,
    step: int,
):
    idx = start_pos + step
    while 0 <= idx < len(sequence):
        frame_value, neighbor_sample_id = sequence[idx]
        if abs(frame_value - current_frame) > temporal_window:
            break

        sample_entry = datadict.get(neighbor_sample_id)
        pose3d = datadict_3d.get(neighbor_sample_id)
        if not isinstance(sample_entry, dict) or pose3d is None:
            idx += step
            continue

        cam_points = sample_entry.get("data", {}).get(cam_name)
        if cam_points is None:
            idx += step
            continue

        cam_points = _as_2d_points(cam_points)
        if joint_idx >= cam_points.shape[1]:
            idx += step
            continue

        point_2d = cam_points[:, joint_idx]
        if not np.isfinite(point_2d).all():
            idx += step
            continue

        anchor_2d = _sample_anchor_2d(
            sample_id=neighbor_sample_id,
            cam_name=cam_name,
            pose3d=_as_pose_matrix(pose3d),
            cameras=cameras,
            com3d_dict=com3d_dict,
            anchor_joint_indices=anchor_joint_indices,
        )
        if anchor_2d is not None:
            return frame_value, point_2d - anchor_2d

        idx += step

    return None


def _flatten_cameras(cameras: Dict) -> Dict:
    if not isinstance(cameras, dict):
        return {}

    flat_cameras = {}
    for key, value in cameras.items():
        if isinstance(value, dict) and (
            "K" in value or "R" in value or "r" in value
        ):
            flat_cameras[str(key)] = value
        elif isinstance(value, dict):
            for cam_name, cam_params in value.items():
                flat_cameras[str(cam_name)] = cam_params

    return flat_cameras


def _as_pose_matrix(pose3d: np.ndarray) -> np.ndarray:
    pose3d = np.asarray(pose3d, dtype=np.float64)
    if pose3d.ndim != 2:
        raise ValueError(f"Expected pose3d to be 2D, got shape {pose3d.shape}")
    if pose3d.shape[0] == 3:
        return pose3d
    if pose3d.shape[1] == 3:
        return pose3d.T
    raise ValueError(f"Could not interpret pose3d with shape {pose3d.shape}")


def _as_2d_points(points_2d: np.ndarray) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2:
        raise ValueError(f"Expected 2D points to be 2D, got shape {points_2d.shape}")
    if points_2d.shape[0] == 2:
        return points_2d
    if points_2d.shape[1] == 2:
        return points_2d.T
    raise ValueError(f"Could not interpret 2D points with shape {points_2d.shape}")


def _get_rotation(cam_params: Dict) -> np.ndarray:
    rotation = cam_params.get("R", cam_params.get("r"))
    if rotation is None:
        raise KeyError("Camera parameters are missing rotation matrix under 'R'/'r'.")
    return np.asarray(rotation, dtype=np.float64)


def _get_translation(cam_params: Dict) -> np.ndarray:
    translation = np.asarray(cam_params["t"], dtype=np.float64)
    if translation.shape == (3,):
        translation = translation[np.newaxis, :]
    elif translation.shape == (3, 1):
        translation = translation.reshape(1, 3)
    return translation


def _get_radial_distortion(cam_params: Dict) -> np.ndarray:
    return np.squeeze(
        np.asarray(cam_params.get("RDistort", np.zeros(3, dtype=np.float64)))
    )


def _get_tangential_distortion(cam_params: Dict) -> np.ndarray:
    return np.squeeze(
        np.asarray(cam_params.get("TDistort", np.zeros(2, dtype=np.float64)))
    )


def _project_to_2d(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    camera_matrix = np.concatenate((rotation, translation), axis=0) @ intrinsic
    projected = np.concatenate(
        (points_3d, np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)),
        axis=1,
    ) @ camera_matrix
    projected[:, :2] = projected[:, :2] / projected[:, 2:3]
    return projected


def _distort_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    radial_distortion: np.ndarray,
    tangential_distortion: np.ndarray,
) -> np.ndarray:
    cx = intrinsic[2, 0]
    cy = intrinsic[2, 1]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    skew = intrinsic[1, 0]

    centered = points - np.array([cx, cy], dtype=np.float64)[np.newaxis, :]
    y_norm = centered[:, 1] / fy
    x_norm = (centered[:, 0] - skew * y_norm) / fx

    r2 = x_norm ** 2 + y_norm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    radial = np.zeros(3, dtype=np.float64)
    radial[: min(2, radial_distortion.size)] = radial_distortion[:2]
    if radial_distortion.size >= 3:
        radial[2] = radial_distortion[2]

    alpha = radial[0] * r2 + radial[1] * r4 + radial[2] * r6

    tangential = np.zeros(2, dtype=np.float64)
    tangential[: min(2, tangential_distortion.size)] = tangential_distortion[:2]
    xy_product = x_norm * y_norm
    dx_tangential = (
        2 * tangential[0] * xy_product + tangential[1] * (r2 + 2 * x_norm ** 2)
    )
    dy_tangential = (
        tangential[0] * (r2 + 2 * y_norm ** 2) + 2 * tangential[1] * xy_product
    )

    normalized = np.stack((x_norm, y_norm), axis=1)
    distorted_normalized = (
        normalized
        + normalized * np.stack((alpha, alpha), axis=1)
        + np.stack((dx_tangential, dy_tangential), axis=1)
    )

    distorted_x = distorted_normalized[:, 0] * fx + cx + skew * distorted_normalized[:, 1]
    distorted_y = distorted_normalized[:, 1] * fy + cy

    return np.stack((distorted_x, distorted_y))


def _camera_center_world(cam_params: Dict) -> np.ndarray | None:
    if cam_params is None:
        return None

    rotation = _get_rotation(cam_params)
    translation = _get_translation(cam_params)
    return (-translation @ np.linalg.inv(rotation)).reshape(3)


def _project_point_to_camera(point_3d: np.ndarray, cam_params: Dict) -> np.ndarray | None:
    rotation = _get_rotation(cam_params)
    translation = _get_translation(cam_params)
    intrinsic = np.asarray(cam_params["K"], dtype=np.float64)

    projected = _project_to_2d(
        np.asarray(point_3d, dtype=np.float64)[np.newaxis, :],
        intrinsic,
        rotation,
        translation,
    )[:, :2]

    radial = _get_radial_distortion(cam_params)
    tangential = _get_tangential_distortion(cam_params)
    if radial.size != 0 or tangential.size != 0:
        projected = _distort_points(projected, intrinsic, radial, tangential).T

    if projected.shape[0] == 0:
        return None
    return projected[0]


def _triangulate_joint_from_other_views(
    sample_entry: Dict,
    cameras: Dict,
    joint_idx: int,
    target_cam: str,
    min_support_views: int,
) -> tuple[np.ndarray | None, int]:
    try:
        from dannce.engine.data import ops
    except ImportError:  # pragma: no cover - exercised only in minimal test envs
        return None, 0

    points = []
    camera_mats = []

    for cam_name, cam_2d in sample_entry.get("data", {}).items():
        if cam_name == target_cam or cam_name not in cameras:
            continue

        points_2d = _as_2d_points(cam_2d)
        if joint_idx >= points_2d.shape[1]:
            continue

        point_2d = points_2d[:, joint_idx]
        if not np.isfinite(point_2d).all():
            continue

        cam_params = cameras[cam_name]
        intrinsic = np.asarray(cam_params["K"], dtype=np.float64)
        rotation = _get_rotation(cam_params)
        translation = _get_translation(cam_params)

        undistorted = ops.unDistortPoints(
            point_2d[np.newaxis, :],
            intrinsic,
            _get_radial_distortion(cam_params),
            _get_tangential_distortion(cam_params),
            rotation,
            translation,
        )
        camera_matrix = ops.camera_matrix(intrinsic, rotation, translation)

        points.append(undistorted)
        camera_mats.append(camera_matrix)

    if len(points) < min_support_views:
        return None, len(points)

    triangulated = ops.triangulate_multi_instance(points, camera_mats)
    point_3d = np.asarray(triangulated[:, 0], dtype=np.float64)
    if not np.isfinite(point_3d).all():
        return None, len(points)

    return point_3d, len(points)


def _build_body_primitives(
    pose3d: np.ndarray,
    joint_names: list[str],
    limbs: list[tuple[int, int]],
    radius_overrides: Dict,
) -> list[Dict]:
    joint_radii = _joint_radii_mm(joint_names, radius_overrides)
    primitives = []

    for joint_idx in range(pose3d.shape[1]):
        point = pose3d[:, joint_idx]
        if not np.isfinite(point).all():
            continue
        primitives.append(
            {
                "type": "sphere",
                "joint_indices": (joint_idx,),
                "center": point,
                "radius": joint_radii[joint_idx],
            }
        )

    for joint_a, joint_b in limbs:
        point_a = pose3d[:, joint_a]
        point_b = pose3d[:, joint_b]
        if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
            continue
        primitives.append(
            {
                "type": "capsule",
                "joint_indices": (joint_a, joint_b),
                "point_a": point_a,
                "point_b": point_b,
                "radius": max(joint_radii[joint_a], joint_radii[joint_b]),
            }
        )

    return primitives


def _joint_radii_mm(joint_names: list[str], overrides: Dict) -> np.ndarray:
    normalized_overrides = {str(key).lower(): float(value) for key, value in overrides.items()}

    radii = []
    for joint_idx, joint_name in enumerate(joint_names):
        override = None
        for lookup_key in (str(joint_idx), joint_name.lower()):
            if lookup_key in normalized_overrides:
                override = normalized_overrides[lookup_key]
                break

        if override is None:
            override = _default_joint_radius_mm(joint_name)

        radii.append(float(override))

    return np.asarray(radii, dtype=np.float64)


def _default_joint_radius_mm(joint_name: str) -> float:
    name = joint_name.lower()
    if any(token in name for token in ("snout", "ear", "head")):
        return 8.0
    if any(
        token in name
        for token in ("spine", "shoulder", "hip", "neck", "body", "tailbase")
    ):
        return 10.0
    if any(
        token in name
        for token in (
            "elbow",
            "knee",
            "wrist",
            "ankle",
            "arm",
            "forelimb",
            "hindlimb",
            "forshd",
            "foreshd",
            "hindshd",
        )
    ):
        return 6.0
    if any(token in name for token in ("hand", "foot", "paw", "tail", "shin", "offset")):
        return 5.0
    return 6.0


def _is_joint_occluded(
    camera_center: np.ndarray,
    target_point: np.ndarray,
    primitives: list[Dict],
    target_joint_idx: int,
) -> bool:
    ray_length = float(np.linalg.norm(target_point - camera_center))
    if ray_length <= 1e-8:
        return False

    for primitive in primitives:
        if target_joint_idx in primitive["joint_indices"]:
            continue

        radius = float(primitive["radius"])
        margin = max(2.0, radius)

        if primitive["type"] == "sphere":
            distance, ray_fraction = _point_to_segment_distance(
                camera_center, target_point, primitive["center"]
            )
        else:
            distance, ray_fraction = _segment_to_segment_distance(
                camera_center,
                target_point,
                primitive["point_a"],
                primitive["point_b"],
            )

        if distance <= radius and (ray_fraction * ray_length) < (ray_length - margin):
            return True

    return False


def _point_to_segment_distance(
    segment_start: np.ndarray, segment_end: np.ndarray, point: np.ndarray
) -> tuple[float, float]:
    direction = segment_end - segment_start
    denom = float(np.dot(direction, direction))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - segment_start)), 1.0

    fraction = float(np.dot(point - segment_start, direction) / denom)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    closest = segment_start + fraction * direction
    return float(np.linalg.norm(point - closest)), fraction


def _segment_to_segment_distance(
    seg1_start: np.ndarray,
    seg1_end: np.ndarray,
    seg2_start: np.ndarray,
    seg2_end: np.ndarray,
) -> tuple[float, float]:
    u = seg1_end - seg1_start
    v = seg2_end - seg2_start
    w = seg1_start - seg2_start

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    small = 1e-8

    if a <= small:
        return _point_to_segment_distance(seg2_start, seg2_end, seg1_start)

    if c <= small:
        return _point_to_segment_distance(seg1_start, seg1_end, seg2_start)

    s_num = denom
    s_den = denom
    t_num = denom
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = e
        t_den = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_den = c
        elif s_num > s_den:
            s_num = s_den
            t_num = e + b
            t_den = c

    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num = -d
            s_den = a
    elif t_num > t_den:
        t_num = t_den
        if (-d + b) < 0.0:
            s_num = 0.0
        elif (-d + b) > a:
            s_num = s_den
        else:
            s_num = -d + b
            s_den = a

    s_fraction = 0.0 if abs(s_num) < small else s_num / s_den
    t_fraction = 0.0 if abs(t_num) < small else t_num / t_den

    closest_seg1 = seg1_start + s_fraction * u
    closest_seg2 = seg2_start + t_fraction * v
    distance = float(np.linalg.norm(closest_seg1 - closest_seg2))

    return distance, float(np.clip(s_fraction, 0.0, 1.0))
