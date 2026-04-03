"""Train the pose-only learned visibility head and export overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dannce.engine.data.io import load_camera_params, load_camnames, load_labels
from dannce.engine.models.nets import initialize_prediction
from dannce.engine.trainer.train_utils import (
    build_visibility_camera_features,
    visibility_dict_to_tensor,
    visibility_tensor_to_dict,
)
from dannce.engine.utils.checkpoint import torch_load_checkpoint
from dannce.engine.utils.visibility import populate_occlusion_visibility

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label3d-file", required=True)
    parser.add_argument("--viddir", required=True)
    parser.add_argument("--npy-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--preview-samples", type=int, default=2)
    parser.add_argument("--scan-samples", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--teacher-mode",
        default="body_side_learned",
        choices=("capsule_raycast", "learned", "body_side_learned"),
    )
    parser.add_argument(
        "--teacher-use-com-filter",
        dest="teacher_use_com_filter",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-teacher-com-filter",
        dest="teacher_use_com_filter",
        action="store_false",
    )
    parser.add_argument("--teacher-com-filter-window", type=int, default=5)
    parser.add_argument("--teacher-com-filter-thresh-px", type=float, default=60.0)
    parser.add_argument(
        "--teacher-com-filter-require-two-sided",
        action="store_true",
        default=True,
    )
    parser.add_argument("--teacher-reproj-guard-px", type=float, default=20.0)
    parser.add_argument(
        "--teacher-reproj-only-from-stored3d",
        action="store_true",
        default=False,
        help="Only let reprojection-based rejection fire when the reference comes from stored 3D labels.",
    )
    parser.add_argument("--teacher-min-support-views", type=int, default=2)
    parser.add_argument(
        "--teacher-temporal-jump-filter",
        dest="teacher_temporal_jump_filter",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-teacher-temporal-jump-filter",
        dest="teacher_temporal_jump_filter",
        action="store_false",
    )
    parser.add_argument("--teacher-temporal-jump-window", type=int, default=5)
    parser.add_argument("--teacher-temporal-jump-thresh-px", type=float, default=75.0)
    parser.add_argument("--teacher-temporal-reference", action="store_true", default=False)
    parser.add_argument("--teacher-temporal-window", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--eval-heldout-every",
        type=int,
        default=1000,
        help="Evaluate and save heldout metrics every N train steps; <=0 disables periodic heldout evaluation.",
    )
    parser.add_argument("--target-train-accuracy", type=float, default=None)
    parser.add_argument("--use-pos-weight", action="store_true", default=True)
    parser.add_argument("--overlay-start-sec", type=float, default=60.0)
    parser.add_argument("--overlay-duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--overlay-sample-hz",
        type=float,
        default=0.0,
        help="Sampling rate for overlay frames; <=0 means original labeled frame rate.",
    )
    parser.add_argument(
        "--overlay-preview-fps",
        type=float,
        default=0.0,
        help="Playback FPS for the overlay video; <=0 means original video FPS.",
    )
    parser.add_argument("--overlay-tile-width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def reshape_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])
    return data.astype(np.float32) - 1.0


def reshape_3d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = np.transpose(np.reshape(data, [data.shape[0], -1, 3]), [0, 2, 1])
    return data.astype(np.float32)


def load_label_arrays(label3d_file: Path):
    camera_order = [str(name) for name in load_camnames(str(label3d_file))]
    labels = load_labels(str(label3d_file))
    camera_params = load_camera_params(str(label3d_file))
    cameras = {0: {cam: camera_params[idx] for idx, cam in enumerate(camera_order)}}

    points_2d = {
        cam: reshape_2d(labels[idx]["data_2d"]) for idx, cam in enumerate(camera_order)
    }
    frames = {
        cam: np.squeeze(labels[idx]["data_frame"]).astype(int)
        for idx, cam in enumerate(camera_order)
    }
    points_3d = reshape_3d(labels[0]["data_3d"])
    return camera_order, cameras, points_2d, frames, points_3d


def build_label_subset(
    label3d_file_or_sample_ids,
    sample_ids_or_params,
    params_or_viddir,
    viddir: str | None = None,
    camera_order: list[str] | None = None,
    cameras: dict | None = None,
    points_2d: dict | None = None,
    frames: dict | None = None,
    points_3d: np.ndarray | None = None,
) -> tuple[dict, dict, dict, list[str]]:
    """Build a small datadict subset for visibility previews.

    Supports both the original call shape:
    ``build_label_subset(label3d_file, sample_ids, params, viddir)``
    and the newer array-based form:
    ``build_label_subset(sample_ids, params, viddir, camera_order, cameras, ...)``.
    """
    if camera_order is None:
        label3d_file = Path(label3d_file_or_sample_ids)
        sample_ids = list(sample_ids_or_params)
        params = params_or_viddir
        if viddir is None:
            raise ValueError("viddir is required when loading labels from file.")
        (
            camera_order,
            cameras,
            points_2d,
            frames,
            points_3d,
        ) = load_label_arrays(label3d_file)
    else:
        sample_ids = list(label3d_file_or_sample_ids)
        params = sample_ids_or_params
        if viddir is None:
            viddir = params_or_viddir

    datadict = {}
    datadict_3d = {}
    for sample_id in sample_ids:
        sample_idx = int(sample_id.split("_", 1)[1])
        datadict[sample_id] = {"data": {}, "frames": {}}
        for cam in camera_order:
            datadict[sample_id]["data"][cam] = points_2d[cam][sample_idx]
            datadict[sample_id]["frames"][cam] = int(frames[cam][sample_idx])
        datadict_3d[sample_id] = points_3d[sample_idx]

    params["experiment"] = {
        0: {
            "camnames": camera_order,
            "viddir": viddir,
            "extension": ".mp4",
            "chunks": {cam: np.array([0], dtype=int) for cam in camera_order},
        }
    }
    params["vid_dir_flag"] = True
    populate_occlusion_visibility(datadict, datadict_3d, cameras, sample_ids, params)
    return datadict, datadict_3d, cameras, camera_order


def load_available_sample_ids(points_3d: np.ndarray) -> list[str]:
    return [f"0_{idx}" for idx in range(int(points_3d.shape[0]))]


def subsample_sample_ids_uniform(sample_ids: list[str], max_count: int) -> list[str]:
    if max_count <= 0 or len(sample_ids) <= max_count:
        return list(sample_ids)
    positions = np.linspace(0, len(sample_ids) - 1, num=max_count, dtype=int)
    return [sample_ids[pos] for pos in positions]


def choose_samples(
    datadict: dict,
    sample_ids: list[str],
    train_count: int,
    preview_count: int,
):
    ranked = []
    for sample_id in sample_ids:
        visibility = datadict[sample_id].get("visibility", {})
        occluded = sum(
            int((~np.asarray(mask, dtype=bool)).sum())
            for mask in visibility.values()
        )
        ranked.append((occluded, sample_id))

    ranked.sort(key=lambda item: (-item[0], int(item[1].split("_", 1)[1])))
    ordered_ids = [sample_id for _, sample_id in ranked]
    train_ids = ordered_ids[:train_count]
    preview_ids = ordered_ids[train_count : train_count + preview_count]
    if len(preview_ids) < preview_count:
        extra = [
            sample_id
            for sample_id in ordered_ids
            if sample_id not in train_ids + preview_ids
        ]
        preview_ids.extend(extra[: preview_count - len(preview_ids)])
    return train_ids, preview_ids, ranked, ordered_ids


def compute_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict:
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    target_bool = targets >= 0.5

    tp = (preds & target_bool).sum().item()
    tn = ((~preds) & (~target_bool)).sum().item()
    fp = (preds & (~target_bool)).sum().item()
    fn = ((~preds) & target_bool).sum().item()
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0
    bce = nn.BCEWithLogitsLoss()(logits, targets).item()

    return {
        "count": int(total),
        "accuracy": float(accuracy),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "bce": float(bce),
        "balanced_accuracy": float(
            (
                (tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                + (tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            )
            / 2.0
        ),
        "pred_visible_rate": float(preds.float().mean().item()),
        "target_visible_rate": float(targets.mean().item()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def predict_visibility_for_ids(
    model,
    sample_ids: list[str],
    datadict: dict,
    datadict_3d: dict,
    cameras: dict,
    camera_order: list[str],
    device: torch.device,
):
    logits_list = []
    targets_list = []
    for sample_id in sample_ids:
        target = visibility_dict_to_tensor(
            datadict[sample_id]["visibility"],
            camera_order,
        ).float()
        coords = (
            torch.from_numpy(np.asarray(datadict_3d[sample_id], dtype=np.float32))
            .unsqueeze(0)
            .to(device)
        )
        camera_features = build_visibility_camera_features(
            [sample_id],
            cameras,
            camera_order,
            device=device,
            dtype=coords.dtype,
        )
        with torch.no_grad():
            logits = model.predict_visibility(coords, camera_features)
        logits_list.append(logits.cpu())
        targets_list.append(target)
        del coords, camera_features, logits

    if len(logits_list) == 0:
        return None, None
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


def save_overlay_previews(
    datadict: dict,
    sample_ids: list[str],
    camera_order: list[str],
    visibility_logits: torch.Tensor,
    visibility_targets: dict,
    viddir: str,
    output_dir: Path,
    threshold: float,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_visibility = torch.sigmoid(visibility_logits) >= threshold
    saved_paths = []
    readers = {cam: imageio.get_reader(str(Path(viddir) / cam / "0.mp4")) for cam in camera_order}

    try:
        for sample_offset, sample_id in enumerate(sample_ids):
            sample_entry = datadict[sample_id]
            for cam_idx, cam_name in enumerate(camera_order):
                frame_idx = int(sample_entry["frames"][cam_name])
                frame = readers[cam_name].get_data(frame_idx)
                points_2d = np.asarray(sample_entry["data"][cam_name], dtype=np.float32)
                pred_mask = predicted_visibility[sample_offset, cam_idx].cpu().numpy()
                target_mask = np.asarray(
                    visibility_targets[cam_name][sample_offset],
                    dtype=bool,
                )

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(frame)
                xs, ys = points_2d[0], points_2d[1]
                finite = np.isfinite(xs) & np.isfinite(ys)
                for joint_idx in np.where(finite)[0]:
                    ax.scatter(
                        xs[joint_idx],
                        ys[joint_idx],
                        s=55,
                        facecolors="none",
                        edgecolors="white" if target_mask[joint_idx] else "yellow",
                        linewidths=1.0,
                    )
                    ax.scatter(
                        xs[joint_idx],
                        ys[joint_idx],
                        s=20,
                        c="lime" if pred_mask[joint_idx] else "crimson",
                        edgecolors="none",
                    )

                ax.set_title(f"{sample_id} {cam_name}")
                ax.axis("off")
                save_path = output_dir / f"{sample_id}_{cam_name}.png"
                fig.tight_layout()
                fig.savefig(save_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                saved_paths.append(str(save_path))
    finally:
        for reader in readers.values():
            reader.close()

    return saved_paths


def draw_marker(frame, point, is_visible):
    x, y = int(round(point[0])), int(round(point[1]))
    if is_visible:
        cv2.circle(frame, (x, y), 7, (60, 220, 60), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), 10, (20, 90, 20), 2, lineType=cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 10, (30, 30, 180), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (x - 7, y - 7), (x + 7, y + 7), (30, 30, 180), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (x - 7, y + 7), (x + 7, y - 7), (30, 30, 180), 2, lineType=cv2.LINE_AA)


def format_time(frame_idx: int, fps: float) -> str:
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def read_frame_at(cap, target_frame: int, state: dict):
    next_frame = state.get("next_frame")
    if next_frame is None or target_frame < next_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        next_frame = target_frame

    frame = None
    while next_frame <= target_frame:
        ok, frame = cap.read()
        if not ok:
            state["next_frame"] = next_frame
            return False, None
        next_frame += 1

    state["next_frame"] = next_frame
    return True, frame


def select_overlay_ids(
    sample_ids: list[str],
    frames: dict,
    camera_order: list[str],
    fps: float,
    start_sec: float,
    duration_sec: float,
    sample_hz: float,
) -> list[str]:
    frame_series = np.asarray(frames[camera_order[0]], dtype=np.int64)
    start_frame = int(round(start_sec * fps))
    end_frame = int(round((start_sec + duration_sec) * fps))
    selected = []
    for sample_id in sample_ids:
        sample_idx = int(sample_id.split("_", 1)[1])
        frame_idx = int(frame_series[sample_idx])
        if start_frame <= frame_idx < end_frame:
            selected.append((frame_idx, sample_id))

    selected.sort(key=lambda item: item[0])
    if sample_hz > 0.0:
        stride = max(1, int(round(fps / sample_hz)))
        selected = selected[::stride]
    return [sample_id for _, sample_id in selected]


def save_overlay_video(
    datadict: dict,
    sample_ids: list[str],
    camera_order: list[str],
    visibility_logits: torch.Tensor,
    viddir: str,
    output_path: Path,
    threshold: float,
    preview_fps: float,
    tile_width: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_visibility = torch.sigmoid(visibility_logits) >= threshold
    video_paths = {cam: Path(viddir) / cam / "0.mp4" for cam in camera_order}
    caps = {cam: cv2.VideoCapture(str(path)) for cam, path in video_paths.items()}
    frame_dir = None

    try:
        if not all(cap.isOpened() for cap in caps.values()):
            missing = [cam for cam, cap in caps.items() if not cap.isOpened()]
            raise RuntimeError(f"Failed to open videos for cameras: {missing}")

        probe_fps = float(caps[camera_order[0]].get(cv2.CAP_PROP_FPS) or 30.0)
        preview_fps = float(preview_fps) if preview_fps > 0 else probe_fps
        ok, probe = read_frame_at(caps[camera_order[0]], int(datadict[sample_ids[0]]["frames"][camera_order[0]]), {"next_frame": None})
        if not ok:
            raise RuntimeError("Failed to read probe frame for overlay video.")

        tile_width = int(tile_width)
        tile_height = int(round(tile_width * probe.shape[0] / probe.shape[1]))
        banner_height = 80
        canvas_size = (tile_width * len(camera_order), tile_height + banner_height)
        frame_dir = Path(
            tempfile.mkdtemp(prefix=f"{output_path.stem}_frames_", dir=str(output_path.parent))
        )

        summary = {
            "preview_fps": preview_fps,
            "n_frames_rendered": len(sample_ids),
            "per_camera": {
                cam: {"used": 0, "excluded": 0, "missing": 0} for cam in camera_order
            },
        }

        cap_states = {cam: {"next_frame": None} for cam in camera_order}
        for render_idx, sample_id in enumerate(sample_ids, start=1):
            sample_entry = datadict[sample_id]
            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            banner = canvas[:banner_height]
            banner[:] = (18, 18, 18)
            frame_idx = int(sample_entry["frames"][camera_order[0]])
            cv2.putText(
                banner,
                "Green=used by trained geometry model  Red=excluded by trained geometry model",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                banner,
                f"frame={frame_idx}  time={format_time(frame_idx, probe_fps)}  sample_id={sample_id}",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
                lineType=cv2.LINE_AA,
            )

            for cam_idx, cam_name in enumerate(camera_order):
                ok, frame = read_frame_at(
                    caps[cam_name],
                    int(sample_entry["frames"][cam_name]),
                    cap_states[cam_name],
                )
                if not ok:
                    frame = np.zeros_like(probe)

                points_2d = np.asarray(sample_entry["data"][cam_name], dtype=np.float32)
                pred_mask = predicted_visibility[render_idx - 1, cam_idx].cpu().numpy()
                finite_mask = np.isfinite(points_2d).all(axis=0)
                used_count = int(np.sum(finite_mask & pred_mask))
                excluded_count = int(np.sum(finite_mask & ~pred_mask))
                missing_count = int(np.sum(~finite_mask))
                summary["per_camera"][cam_name]["used"] += used_count
                summary["per_camera"][cam_name]["excluded"] += excluded_count
                summary["per_camera"][cam_name]["missing"] += missing_count

                for joint_idx in np.where(finite_mask)[0]:
                    draw_marker(frame, points_2d[:, joint_idx], bool(pred_mask[joint_idx]))

                cv2.putText(
                    frame,
                    cam_name,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"used {used_count}  excluded {excluded_count}  missing {missing_count}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

                tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                x0 = cam_idx * tile_width
                canvas[banner_height:, x0 : x0 + tile_width] = tile

            frame_path = frame_dir / f"frame_{render_idx:06d}.png"
            if not cv2.imwrite(str(frame_path), canvas):
                raise RuntimeError(f"Failed to write overlay frame: {frame_path}")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(preview_fps),
                "-i",
                str(frame_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return summary
    finally:
        if frame_dir is not None and frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)
        for cap in caps.values():
            cap.release()


def main(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    label3d_file = Path(args.label3d_file)
    npy_root = Path(args.npy_root) if args.npy_root is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ckpt = torch_load_checkpoint(str(checkpoint_path))
    params = ckpt["params"].copy()
    params["dannce_predict_model"] = str(checkpoint_path)
    params["exclude_occluded_2d"] = True
    params["exclude_occluded_2d_mode"] = str(args.teacher_mode)
    params["learned_visibility_enabled"] = True
    params["exclude_occluded_2d_com_filter"] = bool(args.teacher_use_com_filter)
    params["exclude_occluded_2d_com_filter_window"] = int(
        args.teacher_com_filter_window
    )
    params["exclude_occluded_2d_com_filter_thresh_px"] = float(
        args.teacher_com_filter_thresh_px
    )
    params["exclude_occluded_2d_com_filter_require_two_sided"] = bool(
        args.teacher_com_filter_require_two_sided
    )
    params["exclude_occluded_2d_reproj_guard_px"] = float(args.teacher_reproj_guard_px)
    params["exclude_occluded_2d_reproj_only_from_stored3d"] = bool(
        args.teacher_reproj_only_from_stored3d
    )
    params["exclude_occluded_2d_min_support_views"] = int(args.teacher_min_support_views)
    params["exclude_occluded_2d_temporal_jump_filter"] = bool(
        args.teacher_temporal_jump_filter
    )
    params["exclude_occluded_2d_temporal_jump_window"] = int(
        args.teacher_temporal_jump_window
    )
    params["exclude_occluded_2d_temporal_jump_thresh_px"] = float(
        args.teacher_temporal_jump_thresh_px
    )
    params["exclude_occluded_2d_temporal_reference"] = bool(
        args.teacher_temporal_reference
    )
    params["exclude_occluded_2d_temporal_window"] = int(args.teacher_temporal_window)

    camera_order, cameras, points_2d, frames, points_3d = load_label_arrays(label3d_file)
    available_sample_ids = load_available_sample_ids(points_3d)
    scan_ids = subsample_sample_ids_uniform(
        available_sample_ids, min(args.scan_samples, len(available_sample_ids))
    )
    datadict, datadict_3d, cameras, camera_order = build_label_subset(
        scan_ids,
        params,
        args.viddir,
        camera_order,
        cameras,
        points_2d,
        frames,
        points_3d,
    )
    train_ids, preview_ids, ranked, ordered_ids = choose_samples(
        datadict,
        scan_ids,
        train_count=args.train_samples,
        preview_count=args.preview_samples,
    )
    if len(train_ids) == 0 or len(preview_ids) == 0:
        raise RuntimeError(
            "Could not select enough labeled samples for training and preview."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_prediction(
        params,
        n_cams=len(camera_order),
        device=device,
        model_type="dannce",
    )
    if getattr(model, "visibility_head", None) is None:
        raise RuntimeError("Model was initialized without a visibility head.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.visibility_head.parameters():
        parameter.requires_grad_(True)

    model.eval()
    model.visibility_head.train()
    optimizer = torch.optim.Adam(model.visibility_head.parameters(), lr=args.lr)
    initial_train_targets = torch.cat(
        [
            visibility_dict_to_tensor(datadict[sample_id]["visibility"], camera_order)
            .float()
            for sample_id in train_ids
        ],
        dim=0,
    )
    pos_weight = None
    if args.use_pos_weight:
        positive = float(initial_train_targets.sum().item())
        total = float(initial_train_targets.numel())
        negative = max(total - positive, 1.0)
        positive = max(positive, 1.0)
        pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )

    loss_history = []
    train_metrics_history = []
    heldout_metrics_history = []
    train_order = np.arange(len(train_ids), dtype=np.int64)
    heldout_ids = [sample_id for sample_id in ordered_ids if sample_id not in train_ids]
    for step in range(args.train_steps):
        epoch_step = step % len(train_ids)
        if epoch_step == 0:
            rng.shuffle(train_order)
        sample_id = train_ids[int(train_order[epoch_step])]
        coords = (
            torch.from_numpy(np.asarray(datadict_3d[sample_id], dtype=np.float32))
            .unsqueeze(0)
            .to(device)
        )
        camera_features = build_visibility_camera_features(
            [sample_id],
            cameras,
            camera_order,
            device=device,
            dtype=coords.dtype,
        )
        target = (
            visibility_dict_to_tensor(datadict[sample_id]["visibility"], camera_order)
            .float()
            .to(device)
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model.predict_visibility(coords, camera_features)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_history.append({"step": step + 1, "sample_id": sample_id, "loss": float(loss.item())})
        del target, coords, camera_features, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        should_eval_train = (
            (step + 1) % max(1, args.eval_every) == 0
            or (step + 1) == args.train_steps
        )
        if should_eval_train:
            train_logits, train_targets = predict_visibility_for_ids(
                model,
                train_ids,
                datadict,
                datadict_3d,
                cameras,
                camera_order,
                device,
            )
            train_metrics = compute_binary_metrics(
                train_logits,
                train_targets,
                args.threshold,
            )
            train_metrics["step"] = step + 1
            train_metrics_history.append(train_metrics)
            should_eval_heldout = (
                len(heldout_ids) > 0
                and int(args.eval_heldout_every) > 0
                and (
                    (step + 1) % max(1, int(args.eval_heldout_every)) == 0
                    or (step + 1) == args.train_steps
                )
            )
            if should_eval_heldout:
                heldout_logits, heldout_targets = predict_visibility_for_ids(
                    model,
                    heldout_ids,
                    datadict,
                    datadict_3d,
                    cameras,
                    camera_order,
                    device,
                )
                heldout_metrics = compute_binary_metrics(
                    heldout_logits,
                    heldout_targets,
                    args.threshold,
                )
                heldout_metrics["step"] = step + 1
                heldout_metrics_history.append(heldout_metrics)
            if (
                args.target_train_accuracy is not None
                and train_metrics["accuracy"] >= args.target_train_accuracy
            ):
                break

    train_logits, train_targets = predict_visibility_for_ids(
        model,
        train_ids,
        datadict,
        datadict_3d,
        cameras,
        camera_order,
        device,
    )
    final_train_metrics = compute_binary_metrics(
        train_logits,
        train_targets,
        args.threshold,
    )

    preview_logits_tensor, preview_target_tensor = predict_visibility_for_ids(
        model,
        preview_ids,
        datadict,
        datadict_3d,
        cameras,
        camera_order,
        device,
    )
    heldout_logits, heldout_targets = predict_visibility_for_ids(
        model,
        heldout_ids,
        datadict,
        datadict_3d,
        cameras,
        camera_order,
        device,
    )
    preview_target_dict = visibility_tensor_to_dict(preview_target_tensor, camera_order)
    saved_paths = save_overlay_previews(
        sample_ids=preview_ids,
        datadict=datadict,
        camera_order=camera_order,
        visibility_logits=preview_logits_tensor,
        visibility_targets=preview_target_dict,
        viddir=args.viddir,
        output_dir=output_dir,
        threshold=args.threshold,
    )

    overlay_probe = cv2.VideoCapture(str(Path(args.viddir) / camera_order[0] / "0.mp4"))
    if not overlay_probe.isOpened():
        raise RuntimeError("Failed to open probe video for overlay timing.")
    try:
        overlay_fps = float(overlay_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        overlay_probe.release()

    overlay_ids = select_overlay_ids(
        available_sample_ids,
        frames,
        camera_order,
        fps=overlay_fps,
        start_sec=float(args.overlay_start_sec),
        duration_sec=float(args.overlay_duration_sec),
        sample_hz=float(args.overlay_sample_hz),
    )
    if len(overlay_ids) == 0:
        raise RuntimeError("No labeled samples matched the requested overlay time window.")
    overlay_datadict, overlay_datadict_3d, _, _ = build_label_subset(
        overlay_ids,
        params,
        args.viddir,
        camera_order,
        cameras,
        points_2d,
        frames,
        points_3d,
    )
    overlay_logits, overlay_targets = predict_visibility_for_ids(
        model,
        overlay_ids,
        overlay_datadict,
        overlay_datadict_3d,
        cameras,
        camera_order,
        device,
    )
    overlay_metrics = compute_binary_metrics(
        overlay_logits,
        overlay_targets,
        args.threshold,
    )
    overlay_video_path = output_dir / "trained_visibility_overlay.mp4"
    overlay_summary = save_overlay_video(
        datadict=overlay_datadict,
        sample_ids=overlay_ids,
        camera_order=camera_order,
        visibility_logits=overlay_logits,
        viddir=args.viddir,
        output_path=overlay_video_path,
        threshold=args.threshold,
        preview_fps=float(args.overlay_preview_fps),
        tile_width=int(args.overlay_tile_width),
    )
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "teacher_mode": str(args.teacher_mode),
            "teacher_reproj_only_from_stored3d": bool(
                args.teacher_reproj_only_from_stored3d
            ),
            "threshold": float(args.threshold),
            "camera_order": camera_order,
            "state_dict": model.visibility_head.state_dict(),
        },
        output_dir / "visibility_head.pt",
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "label3d_file": str(label3d_file),
        "npy_root": str(npy_root) if npy_root is not None else None,
        "camera_order": camera_order,
        "teacher_mode": str(args.teacher_mode),
        "train_ids": train_ids,
        "preview_ids": preview_ids,
        "heldout_ids": heldout_ids,
        "overlay_ids_head": overlay_ids[:10],
        "overlay_sample_count": len(overlay_ids),
        "train_steps": args.train_steps,
        "train_sample_count": len(train_ids),
        "scan_sample_count": len(scan_ids),
        "available_label_sample_count": len(available_sample_ids),
        "pose_only_visibility": True,
        "teacher_use_com_filter": bool(args.teacher_use_com_filter),
        "teacher_reproj_only_from_stored3d": bool(
            args.teacher_reproj_only_from_stored3d
        ),
        "teacher_com_filter_window": int(args.teacher_com_filter_window),
        "teacher_com_filter_thresh_px": float(args.teacher_com_filter_thresh_px),
        "use_pos_weight": bool(args.use_pos_weight),
        "pos_weight": (float(pos_weight.item()) if pos_weight is not None else None),
        "ranked_samples": [
            {"sample_id": sample_id, "occluded_joints": occluded}
            for occluded, sample_id in ranked
        ],
        "loss_history_head": loss_history[:10],
        "loss_history_tail": loss_history[-10:],
        "train_metrics": final_train_metrics,
        "train_metrics_history": train_metrics_history,
        "train_metrics_history_tail": train_metrics_history[-10:],
        "preview_metrics": compute_binary_metrics(
            preview_logits_tensor,
            preview_target_tensor,
            args.threshold,
        ),
        "heldout_metrics": (
            compute_binary_metrics(heldout_logits, heldout_targets, args.threshold)
            if heldout_logits is not None
            else None
        ),
        "heldout_metrics_history": heldout_metrics_history,
        "heldout_metrics_history_tail": heldout_metrics_history[-10:],
        "overlay_metrics": overlay_metrics,
        "overlay_video": str(overlay_video_path),
        "overlay_video_summary": overlay_summary,
        "saved_paths": saved_paths,
        "visibility_head_checkpoint": str(output_dir / "visibility_head.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
