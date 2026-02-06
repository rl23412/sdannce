#!/usr/bin/env python3
"""
Reproject 3D labels to 2D and render overlay videos per camera.

This script:
1) Loads a Label3D (*_Label3D_dannce.mat) file
2) Projects 3D points to 2D using camera parameters
3) Overlays the projected skeleton on the original videos
4) Writes per-camera MP4 videos

It can run on a single label file or search a root directory for many labels.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception as exc:
    raise SystemExit(
        "Pillow is required. Install with: python -m pip install pillow"
    ) from exc

try:
    import imageio
except Exception as exc:
    raise SystemExit(
        "imageio is required. Install with: python -m pip install imageio imageio-ffmpeg"
    ) from exc

# Ensure repo root is on sys.path so local imports work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dannce.engine.data.io import load_camnames, load_camera_params, load_labels
from dannce.engine.skeletons.utils import load_body_profile


def project_to_2d_numpy(pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using DANNCE's numpy convention.

    Args:
        pts: [N, 3] 3D points
        K: [3, 3] intrinsics
        R: [3, 3] rotation
        t: [1, 3] translation (row-wise)

    Returns:
        [N, 2] pixel coordinates (undistorted)
    """
    if t.ndim == 1:
        t = t.reshape(1, 3)
    elif t.shape == (3, 1):
        t = t.reshape(1, 3)

    M = np.concatenate((R, t), axis=0) @ K
    pts_h = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    proj = pts_h @ M
    proj[:, :2] = proj[:, :2] / proj[:, 2:3]
    return proj[:, :2]


def distort_points_numpy(
    points: np.ndarray,
    intrinsic_matrix: np.ndarray,
    radial_distortion: np.ndarray,
    tangential_distortion: np.ndarray,
) -> np.ndarray:
    """Apply radial/tangential distortion (ported from DANNCE utils)."""
    cx = intrinsic_matrix[2, 0]
    cy = intrinsic_matrix[2, 1]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    skew = intrinsic_matrix[1, 0]

    center = np.array([cx, cy])
    centered = points - center[np.newaxis, :]

    y_norm = centered[:, 1] / fy
    x_norm = (centered[:, 0] - skew * y_norm) / fx

    r2 = x_norm**2 + y_norm**2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radial_distortion[:2]
    k[2] = radial_distortion[2] if len(radial_distortion) >= 3 else 0
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    p = tangential_distortion
    xy_product = x_norm * y_norm
    dx_tan = 2 * p[0] * xy_product + p[1] * (r2 + 2 * x_norm**2)
    dy_tan = p[0] * (r2 + 2 * y_norm**2) + 2 * p[1] * xy_product

    normalized = np.stack((x_norm, y_norm)).T
    distorted_norm = (
        normalized
        + normalized * np.array([alpha, alpha]).T
        + np.stack((dx_tan, dy_tan)).T
    )

    distorted_x = distorted_norm[:, 0] * fx + cx + skew * distorted_norm[:, 1]
    distorted_y = distorted_norm[:, 1] * fy + cy
    return np.stack((distorted_x, distorted_y)).T


def infer_skeleton_name(n_keypoints: int) -> Optional[str]:
    """Infer skeleton profile from keypoint count."""
    candidates = ["mouse19", "mouse14", "mouse22", "rat23", "rat16", "rat7m"]
    matches = []
    for name in candidates:
        try:
            profile = load_body_profile(name)
            if len(profile["joint_names"]) == n_keypoints:
                matches.append(name)
        except Exception:
            continue

    if len(matches) == 1:
        return matches[0]
    return None


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    points_2d: np.ndarray,
    connectivity: List[List[int]],
    point_radius: int,
    line_width: int,
    color: Tuple[int, int, int],
    show_index: bool,
) -> None:
    """Draw a skeleton onto an ImageDraw canvas."""
    # Draw limbs
    for i, j in connectivity:
        if i >= points_2d.shape[0] or j >= points_2d.shape[0]:
            continue
        xi, yi = points_2d[i]
        xj, yj = points_2d[j]
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(xj) and np.isfinite(yj)):
            continue
        draw.line([(xi, yi), (xj, yj)], fill=color, width=line_width)

    # Draw keypoints
    for k, (x, y) in enumerate(points_2d):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        left = x - point_radius
        top = y - point_radius
        right = x + point_radius
        bottom = y + point_radius
        draw.ellipse([left, top, right, bottom], outline=(0, 0, 0), fill=color)
        if show_index:
            draw.text((x + 4, y + 4), str(k), fill=(255, 255, 255))


def overlay_skeletons(
    frame: np.ndarray,
    proj_2d: np.ndarray,
    connectivity: List[List[int]],
    labels_2d: Optional[np.ndarray],
    show_index: bool,
) -> np.ndarray:
    """Overlay projected 3D (red) and optional 2D labels (green) on a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Projected 3D -> 2D (red)
    draw_skeleton(
        draw=draw,
        points_2d=proj_2d,
        connectivity=connectivity,
        point_radius=3,
        line_width=2,
        color=(255, 0, 0),
        show_index=show_index,
    )

    # 2D labels (green) if provided
    if labels_2d is not None:
        draw_skeleton(
            draw=draw,
            points_2d=labels_2d,
            connectivity=connectivity,
            point_radius=3,
            line_width=2,
            color=(0, 255, 0),
            show_index=show_index,
        )

    return np.asarray(img)


def find_video_file(cam_dir: str) -> Optional[str]:
    """Find the first mp4 in a camera directory."""
    if not os.path.isdir(cam_dir):
        return None
    mp4s = sorted(glob.glob(os.path.join(cam_dir, "*.mp4")))
    return mp4s[0] if mp4s else None


def load_label3d(label_path: str):
    camnames = load_camnames(label_path)
    if camnames is None:
        raise RuntimeError(f"Could not read camera names from {label_path}")
    labels = load_labels(label_path)
    if not labels:
        raise RuntimeError(f"No labelData entries in {label_path}")
    params = load_camera_params(label_path)
    cameras = {name: params[i] for i, name in enumerate(camnames)}
    return camnames, labels, cameras


def reshape_3d(data_3d: np.ndarray) -> np.ndarray:
    """Return 3D data as [n_frames, n_keypoints, 3]."""
    if data_3d.ndim == 3:
        if data_3d.shape[1] == 3:
            return data_3d.transpose(0, 2, 1)
        return data_3d

    if data_3d.ndim == 2:
        n_keypoints = data_3d.shape[1] // 3
        return data_3d.reshape(data_3d.shape[0], n_keypoints, 3)

    raise ValueError(f"Unexpected data_3d shape: {data_3d.shape}")


def iter_label_files(labels_root: Optional[str], label3d: Optional[str]) -> List[str]:
    if label3d:
        return [label3d]
    if not labels_root:
        return []
    pattern = os.path.join(labels_root, "**", "*Label3D_dannce.mat")
    return sorted(glob.glob(pattern, recursive=True))


def process_label_file(
    label_path: str,
    output_root: str,
    videos_dir: Optional[str],
    skeleton_name: Optional[str],
    apply_distortion: bool,
    pixel_offset: float,
    overlay_2d_labels: bool,
    labels_2d_offset: float,
    start_frame: int,
    end_frame: Optional[int],
    frame_step: int,
    max_frames: Optional[int],
    camera_filter: Optional[List[str]],
    show_index: bool,
    fps_override: Optional[float],
    macro_block_size: int,
    first_seconds: Optional[float],
) -> None:
    camnames, labels, cameras = load_label3d(label_path)

    label_dir = os.path.dirname(label_path)
    videos_dir = videos_dir or os.path.join(label_dir, "videos")
    vid_name = os.path.basename(label_dir)

    data_3d = labels[0]["data_3d"]
    data_frame = labels[0]["data_frame"].reshape(-1)
    data_3d = reshape_3d(data_3d)

    n_frames, n_keypoints, _ = data_3d.shape
    if skeleton_name is None:
        skeleton_name = infer_skeleton_name(n_keypoints)
    if skeleton_name is None:
        raise RuntimeError(
            f"Could not infer skeleton for {n_keypoints} keypoints. "
            "Pass --skeleton explicitly."
        )

    skeleton = load_body_profile(skeleton_name)
    connectivity = skeleton["limbs"]

    if camera_filter:
        camnames = [c for c in camnames if c in camera_filter]
        if not camnames:
            raise RuntimeError("No cameras matched --cameras filter")

    # Select frames using data_frame indices
    frame_ids = data_frame
    if end_frame is None:
        end_frame = int(frame_ids.max())
    valid_idx = np.where((frame_ids >= start_frame) & (frame_ids <= end_frame))[0]
    if frame_step > 1:
        valid_idx = valid_idx[::frame_step]
    if max_frames is not None:
        valid_idx = valid_idx[:max_frames]

    if valid_idx.size == 0:
        print(f"No frames in range for {label_path}")
        return

    output_dir = os.path.join(output_root, vid_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Processing {label_path} ===")
    print(f"Videos dir: {videos_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Skeleton: {skeleton_name} ({n_keypoints} keypoints)")
    print(f"Frames: {valid_idx.size} (start={start_frame}, end={end_frame}, step={frame_step})")

    # Pre-load 2D label data per camera (if requested)
    labels_2d_by_cam = {}
    if overlay_2d_labels:
        for cam_idx, cam_name in enumerate(camnames):
            data_2d = labels[cam_idx].get("data_2d")
            if data_2d is None:
                labels_2d_by_cam[cam_name] = None
                continue
            try:
                pts2d = data_2d.reshape(data_2d.shape[0], n_keypoints, 2)
                labels_2d_by_cam[cam_name] = pts2d
            except Exception:
                labels_2d_by_cam[cam_name] = None

    for cam_name in camnames:
        cam_dir = os.path.join(videos_dir, cam_name)
        video_path = find_video_file(cam_dir)
        if not video_path:
            print(f"  - {cam_name}: video not found under {cam_dir}")
            continue

        cam_params = cameras[cam_name]
        K = np.array(cam_params["K"])
        R = np.array(cam_params.get("R", cam_params.get("r")))
        t = np.array(cam_params["t"])

        rd = cam_params.get("RDistort")
        td = cam_params.get("TDistort")

        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = fps_override or meta.get("fps", 30)

        out_path = os.path.join(output_dir, f"{cam_name}_reprojected_2d.mp4")
        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264", macro_block_size=macro_block_size
        )

        print(f"  - {cam_name}: {os.path.basename(video_path)} -> {out_path}")

        processed = 0
        try:
            # If first_seconds is set, cap frames based on video fps
            local_valid_idx = valid_idx
            if first_seconds is not None:
                max_local_frames = int(round(fps * first_seconds))
                local_valid_idx = valid_idx[:max_local_frames]

            for idx in local_valid_idx:
                frame_id = int(frame_ids[idx])
                try:
                    frame = reader.get_data(frame_id)
                except Exception as exc:
                    print(f"    ! Failed reading frame {frame_id}: {exc}")
                    break

                pts_3d = data_3d[idx]
                proj = project_to_2d_numpy(pts_3d, K, R, t)

                if apply_distortion and rd is not None and td is not None:
                    proj = distort_points_numpy(
                        proj,
                        K,
                        np.squeeze(rd),
                        np.squeeze(td),
                    )

                if pixel_offset != 0:
                    proj = proj + pixel_offset

                labels_2d = None
                if overlay_2d_labels:
                    pts2d_all = labels_2d_by_cam.get(cam_name)
                    if pts2d_all is not None and idx < pts2d_all.shape[0]:
                        labels_2d = pts2d_all[idx].copy()
                        if labels_2d_offset != 0:
                            labels_2d = labels_2d + labels_2d_offset

                frame_overlay = overlay_skeletons(
                    frame=frame,
                    proj_2d=proj,
                    connectivity=connectivity,
                    labels_2d=labels_2d,
                    show_index=show_index,
                )
                writer.append_data(frame_overlay)

                processed += 1
                if processed % 500 == 0:
                    print(f"    ... {processed} frames")
        finally:
            writer.close()
            reader.close()

        print(f"    Done: {processed} frames")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproject 3D labels to 2D overlay videos per camera",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label3d", help="Path to *_Label3D_dannce.mat")
    group.add_argument("--labels-root", help="Root directory to search for label files")

    parser.add_argument(
        "--output-dir",
        default="./reprojected_2d",
        help="Output root directory for videos",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Override videos directory (defaults to <label_dir>/videos)",
    )
    parser.add_argument(
        "--skeleton",
        default=None,
        help="Skeleton profile name (e.g., mouse19, mouse14)",
    )
    parser.add_argument(
        "--no-distortion",
        action="store_true",
        help="Disable lens distortion during projection",
    )
    parser.add_argument(
        "--pixel-offset",
        type=float,
        default=0.0,
        help="Add a constant offset to projected pixel coordinates",
    )
    parser.add_argument(
        "--overlay-2d-labels",
        action="store_true",
        help="Overlay 2D labels in green on top of the video",
    )
    parser.add_argument(
        "--labels-2d-offset",
        type=float,
        default=-1.0,
        help="Offset applied to 2D label coordinates (default: -1 for MATLAB->0-based)",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument(
        "--end-frame", type=int, default=None, help="End frame index (inclusive)"
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit frames processed per camera"
    )
    parser.add_argument(
        "--first-seconds",
        type=float,
        default=None,
        help="Process only the first N seconds (overrides --max-frames per camera)",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera list (e.g., Camera1,Camera2)",
    )
    parser.add_argument(
        "--show-index",
        action="store_true",
        help="Draw keypoint indices",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps (default: video fps)",
    )
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=1,
        help=(
            "FFmpeg macro block size. Use 1 to avoid resizing (default), "
            "or 16 for broader compatibility."
        ),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="When using --labels-root, limit number of label files processed",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    label_files = iter_label_files(args.labels_root, args.label3d)
    if args.max_videos is not None:
        label_files = label_files[: args.max_videos]

    if not label_files:
        print("No label files found.")
        return 1

    camera_filter = None
    if args.cameras:
        camera_filter = [c.strip() for c in args.cameras.split(",") if c.strip()]

    for label_path in label_files:
        try:
            process_label_file(
                label_path=label_path,
                output_root=args.output_dir,
                videos_dir=args.videos_dir,
                skeleton_name=args.skeleton,
                apply_distortion=not args.no_distortion,
                pixel_offset=args.pixel_offset,
                overlay_2d_labels=args.overlay_2d_labels,
                labels_2d_offset=args.labels_2d_offset,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                camera_filter=camera_filter,
                show_index=args.show_index,
                fps_override=args.fps,
                macro_block_size=args.macro_block_size,
                first_seconds=args.first_seconds,
            )
        except Exception as exc:
            print(f"ERROR processing {label_path}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
