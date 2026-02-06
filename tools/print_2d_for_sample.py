import os
import sys
from typing import Optional, List

import numpy as np
from scipy import io as sio


def _try_load_camnames(path: str) -> List[str]:
    try:
        m = sio.loadmat(path)
        if "camnames" in m:
            names = m["camnames"]
            try:
                # Two common shapes: [N,1] of objects vs [1,N] cell
                if names.ndim == 2 and names.shape[0] == 1:
                    return [n[0] for n in names[0]]
                else:
                    return [n[0][0] for n in names]
            except Exception:
                pass
    except Exception:
        pass
    return ["Camera1", "Camera2"]


def _load_label3d_via_scipy(path: str):
    m = sio.loadmat(path)
    ld = m["labelData"]
    # Mirror logic from io.load_label3d_data for scipy case
    dataset = [f[0] for f in ld]
    data = []
    for rec in dataset:
        d_ = {}
        for nm in rec.dtype.names:
            d_[nm] = rec[nm][0, 0]
        data.append(d_)
    return data


def extract_2d_for_frame(label3d_path: str, target_frame: int):
    # Ensure project root is importable
    sys.path.append(os.path.abspath("."))

    # Load camera names and labelData structs via scipy only
    camnames = _try_load_camnames(label3d_path)
    labels = _load_label3d_via_scipy(label3d_path)

    # Build framedict and ddict exactly like prepare_data (transpose + minus 1)
    framedict = {}
    ddict = {}
    for i, cam in enumerate(camnames):
        label = labels[i]
        frames = np.squeeze(label["data_frame"]).astype(int)
        data = label["data_2d"]

        if len(data.shape) == 2:
            # [T, 2*n_kpts] -> [T, 2, n_kpts]
            data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])

        # MATLAB -> 0-based pixel coords
        data = data - 1

        framedict[cam] = frames
        ddict[cam] = data  # shape [T, 2, n_kpts]

    # Find index where Camera1 has the requested frame (and ensure Camera2 matches if present)
    idxs = np.where(framedict[camnames[0]] == target_frame)[0]
    if idxs.size == 0:
        print(f"❌ Frame {target_frame} not found for {camnames[0]}.")
        print(f"  First 5 frames: {framedict[camnames[0]][:5]}")
        return

    sel_idx = None
    for i in idxs:
        ok = True
        for cam in camnames:
            if target_frame not in framedict[cam]:
                ok = False
                break
        if ok:
            sel_idx = i
            break

    if sel_idx is None:
        sel_idx = int(idxs[0])

    print(f"Using index {sel_idx} for frame {target_frame}")
    for cam in camnames[:2]:
        arr = ddict[cam][sel_idx]  # [2, n_kpts]
        print(f"  {cam} 2D shape: {arr.shape}")
        print(f"  {cam} 2D coords (2 x n_kpts):\n{arr}")
        print(f"  {cam} 2D coords as (n_kpts x 2):\n{arr.T}")


def extract_2d_for_sampleid(label3d_path: str, target_sample_id: int):
    sys.path.append(os.path.abspath("."))

    camnames = _try_load_camnames(label3d_path)
    labels = _load_label3d_via_scipy(label3d_path)

    # Build framedict and ddict as in prepare_data
    framedict = {}
    ddict = {}
    for i, cam in enumerate(camnames):
        label = labels[i]
        frames = np.squeeze(label["data_frame"]).astype(int)
        data = label["data_2d"]
        if len(data.shape) == 2:
            data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])
        data = data - 1
        framedict[cam] = frames
        ddict[cam] = data

    # Obtain the samples array exactly as training does
    samples = np.squeeze(labels[0]["data_sampleID"]).astype(int)
    idxs = np.where(samples == target_sample_id)[0]
    if idxs.size == 0:
        print(f"❌ sampleID {target_sample_id} not found. First 5 sampleIDs: {samples[:5]}")
        return
    sel_idx = int(idxs[0])
    print(f"Using dataset index {sel_idx} for sampleID {target_sample_id}")
    # Also show the per-camera data_frame stored for this index
    for cam in camnames[:2]:
        print(f"  {cam} data_frame at this index: {framedict[cam][sel_idx]}")

    for cam in camnames[:2]:
        arr = ddict[cam][sel_idx]
        print(f"  {cam} 2D shape: {arr.shape}")
        print(f"  {cam} 2D coords (2 x n_kpts):\n{arr}")
        print(f"  {cam} 2D coords as (n_kpts x 2):\n{arr.T}")


def inspect_transpose_formats(label3d_path: str, frame_index: int, cam_index: int = 0, raw_im_w: int = 1280, raw_im_h: int = 720):
    """Inspect a single frame row under two layout assumptions:
    A) row = [x1, y1, x2, y2, ...]  -> reshape(-1,2).T => [2, n_kpts]
    B) row = [x1..xn, y1..yn]       -> stack(row[:n], row[n:]) => [2, n_kpts]
    Prints both versions and simple bounds checks.
    """
    camnames = _try_load_camnames(label3d_path)
    labels = _load_label3d_via_scipy(label3d_path)

    label = labels[cam_index]
    data = label["data_2d"]
    frames = np.squeeze(label["data_frame"]).astype(int)

    # Select target index by frame number
    idxs = np.where(frames == frame_index)[0]
    if idxs.size == 0:
        print(f"No frame {frame_index} for {camnames[cam_index]}.")
        return
    i = int(idxs[0])

    row = np.squeeze(data[i])  # shape (2*n_kpts,)
    n = row.shape[0] // 2

    print(f"Inspecting {camnames[cam_index]} frame={frame_index}, row length={row.shape[0]} (2*n_kpts)")
    print("Raw first 28 values:")
    with np.printoptions(precision=7, suppress=False):
        print(row[:28])

    # A) pair-based
    arr_pairs = np.reshape(row, (-1, 2)).T - 1  # [2, n_kpts], subtract 1 for MATLAB indexing
    # B) half-split
    arr_half = np.stack([row[:n], row[n:]], axis=0) - 1

    def bounds_info(arr):
        x, y = arr[0], arr[1]
        valid = ~np.isnan(x) & ~np.isnan(y)
        x_valid, y_valid = x[valid], y[valid]
        out_x = ((x_valid < 0) | (x_valid >= raw_im_w)).sum()
        out_y = ((y_valid < 0) | (y_valid >= raw_im_h)).sum()
        return x_valid.min() if x_valid.size else np.nan, x_valid.max() if x_valid.size else np.nan, \
               y_valid.min() if y_valid.size else np.nan, y_valid.max() if y_valid.size else np.nan, \
               int(out_x), int(out_y)

    xmn, xmx, ymn, ymx, ox, oy = bounds_info(arr_pairs)
    print("\nAssumption A (pairs: [x1,y1,x2,y2,...] -> reshape(-1,2).T), minus 1:")
    print(arr_pairs)
    print(f"x in [{xmn:.1f},{xmx:.1f}], y in [{ymn:.1f},{ymx:.1f}], out_of_bounds x={ox}, y={oy}")

    xmn, xmx, ymn, ymx, ox, oy = bounds_info(arr_half)
    print("\nAssumption B (half-split: [x1..xn,y1..yn] -> stack), minus 1:")
    print(arr_half)
    print(f"x in [{xmn:.1f},{xmx:.1f}], y in [{ymn:.1f},{ymx:.1f}], out_of_bounds x={ox}, y={oy}")


if __name__ == "__main__":
    label3d_path = r"C:\\Users\\Runda\\Downloads\\vid1_Label3D_dannce (11).mat"
    print("--- Using data_frame match ---")
    extract_2d_for_frame(label3d_path, target_frame=1404)
    print("\n--- Using data_sampleID match (dataset way) ---")
    extract_2d_for_sampleid(label3d_path, target_sample_id=1404)
    print("\n--- Transpose inspection (frame 1, Camera1) ---")
    inspect_transpose_formats(label3d_path, frame_index=1404, cam_index=0)


