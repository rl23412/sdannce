import numpy as np
import os
from datetime import datetime
import torch

import dannce.engine.models.loss as custom_losses
import dannce.engine.models.metrics as custom_metrics
from dannce.engine.utils.projection import (
    project_to_2d,
    distortPoints,
    project_to_2d_torch,
    distort_points_torch,
)


def project_batch_with_experiment_aware_cameras(
    kpts_pred, sample_ids, flat_cameras, cam_name, loss_params, debug_info=None
):
    """
    Project 3D keypoints to 2D using experiment-specific camera parameters.
    
    When a batch contains samples from multiple experiments, this function:
    1. Groups samples by their experiment ID
    2. Projects each group using the correct camera parameters
    3. Reassembles the results in the original batch order
    
    Args:
        kpts_pred (torch.Tensor): [B, 3, N] 3D keypoints predictions
        sample_ids (list): List of sample IDs for each batch item
        flat_cameras (dict): Dictionary of camera parameters with experiment prefixes
        cam_name (str): Base camera name (e.g., 'Camera1')
        loss_params (dict): Loss parameters including distortion settings
        debug_info (dict): Optional debug information
        
    Returns:
        torch.Tensor: [B, 2, N] projected 2D points in original batch order
    """
    batch_size, _, n_keypoints = kpts_pred.shape
    device = kpts_pred.device
    dtype = kpts_pred.dtype
    
    # If no sample IDs provided, fall back to single camera projection
    if sample_ids is None or len(sample_ids) != batch_size:
        # Try to find any matching camera parameters
        matching_cam_params = None
        for cam_param_name, cam_params in flat_cameras.items():
            if cam_param_name == cam_name or cam_param_name.endswith(f"_{cam_name}"):
                matching_cam_params = cam_params
                break
        
        if matching_cam_params is None:
            # Return NaN tensor if no camera params found
            return torch.full((batch_size, 2, n_keypoints), float('nan'), device=device, dtype=dtype)
        
        # Project entire batch with single camera params
        return _project_and_distort_single(kpts_pred, matching_cam_params, loss_params)
    
    # Group samples by experiment ID
    experiment_groups = {}
    for i, sample_id in enumerate(sample_ids):
        if isinstance(sample_id, str) and '_' in sample_id:
            exp_id = sample_id.split('_')[0]
            if exp_id.isdigit():
                if exp_id not in experiment_groups:
                    experiment_groups[exp_id] = []
                experiment_groups[exp_id].append(i)
            else:
                # Sample ID doesn't follow expected format
                if 'unknown' not in experiment_groups:
                    experiment_groups['unknown'] = []
                experiment_groups['unknown'].append(i)
        else:
            # Sample ID doesn't follow expected format
            if 'unknown' not in experiment_groups:
                experiment_groups['unknown'] = []
            experiment_groups['unknown'].append(i)
    
    # Initialize output tensor
    proj_2d_all = torch.full((batch_size, 2, n_keypoints), float('nan'), device=device, dtype=dtype)
    
    # Debug header for this camera
    if debug_info and debug_info.get('debug_batch_counter', 999) < 3 and len(experiment_groups) > 0:
        print(f"\n   🎥 Projecting for camera: {cam_name}", flush=True)
    
    # Process each experiment group
    for exp_id, indices in experiment_groups.items():
        # Find camera parameters for this experiment
        cam_param_name = None
        if exp_id == 'unknown':
            # Try to find camera without experiment prefix
            cam_params = flat_cameras.get(cam_name)
            if cam_params is not None:
                cam_param_name = cam_name
        else:
            # Look for experiment-specific camera
            expected_name = f"{exp_id}_{cam_name}"
            cam_params = flat_cameras.get(expected_name)
            if cam_params is not None:
                cam_param_name = expected_name
        
        if cam_params is None:
            # Try fallback strategies
            for candidate_name, params in flat_cameras.items():
                if '_' in candidate_name and candidate_name.split('_')[0].isdigit():
                    base_cam = '_'.join(candidate_name.split('_')[1:])
                    if base_cam == cam_name:
                        cam_params = params
                        cam_param_name = candidate_name
                        if debug_info and debug_info.get('debug_batch_counter', 999) < 3:
                            print(f"   Using fallback camera '{candidate_name}' for experiment '{exp_id}'", flush=True)
                        break
        
        if cam_params is not None:
            # Extract keypoints for this experiment group
            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            kpts_group = kpts_pred[indices_tensor]  # [G, 3, N] where G is group size
            
            # Debug: Print which camera is used for which samples
            if debug_info and debug_info.get('debug_batch_counter', 999) < 3:
                sample_list = [f"{i}" for i in indices[:3]]  # Show first 3 indices
                if len(indices) > 3:
                    sample_list.append(f"...{len(indices)} total")
                print(f"   📷 Exp {exp_id}: Using '{cam_param_name}' for {cam_name}, samples {sample_list}", flush=True)
                if 't' in cam_params:
                    t_vec = cam_params['t'].flatten()[:3] if hasattr(cam_params['t'], 'flatten') else cam_params['t'][:3]
                    print(f"      Position: {t_vec}", flush=True)
            
            # Project this group
            proj_2d_group = _project_and_distort_single(kpts_group, cam_params, loss_params)
            
            # Place results back in original positions
            proj_2d_all[indices_tensor] = proj_2d_group
        elif debug_info and debug_info.get('debug_batch_counter', 999) < 3:
            print(f"   ⚠️  Warning: No camera parameters found for experiment '{exp_id}', camera '{cam_name}'", flush=True)
    
    return proj_2d_all


def _project_and_distort_single(kpts_batch, cam_params, loss_params):
    """
    Helper function to project and optionally distort a batch of keypoints.
    
    Args:
        kpts_batch (torch.Tensor): [B, 3, N] 3D keypoints
        cam_params (dict): Camera parameters (K, R/r, t, RDistort, TDistort)
        loss_params (dict): Loss parameters including distortion settings
        
    Returns:
        torch.Tensor: [B, 2, N] projected 2D points
    """
    device = kpts_batch.device
    dtype = kpts_batch.dtype
    
    # Convert camera parameters to torch
    K_t = torch.as_tensor(cam_params['K'], dtype=dtype, device=device)
    R_t = torch.as_tensor(cam_params.get('R', cam_params.get('r')), dtype=dtype, device=device)
    
    t_np = cam_params['t']
    if isinstance(t_np, np.ndarray):
        t_t = torch.from_numpy(t_np).to(device=device, dtype=dtype)
    else:
        t_t = torch.as_tensor(t_np, dtype=dtype, device=device)
    
    # Ensure correct shape for translation vector
    if t_t.dim() == 2 and t_t.shape[1] == 1:
        t_t = t_t.view(1, 3)
    if t_t.dim() == 1:
        t_t = t_t.view(1, 3)
    
    # Check for extreme 3D coordinates that might cause projection issues
    if loss_params.get('check_3d_bounds', True):
        kpts_max = kpts_batch.abs().max().item()
        if kpts_max > 10000:  # 10 meters in mm
            print(f"   ⚠️  Warning: Extreme 3D coordinates detected (max abs: {kpts_max:.2f})", flush=True)
    
    # Project to 2D
    proj_2d = project_to_2d_torch(kpts_batch, K_t, R_t, t_t)
    
    # Check for extreme projection values
    if loss_params.get('check_projection_bounds', True):
        proj_max = proj_2d.abs().max().item()
        if proj_max > 50000:  # Way outside any reasonable image bounds
            print(f"   ⚠️  Warning: Extreme 2D projection detected (max abs: {proj_max:.2f})", flush=True)
            # Optional: clip to reasonable bounds
            if loss_params.get('clip_projections', True):
                max_coord = loss_params.get('max_projection_coord', 10000.0)
                proj_2d = torch.clamp(proj_2d, min=-max_coord, max=max_coord)
                print(f"   🔧 Clipped projections to ±{max_coord}", flush=True)
    
    # Apply distortion if available and requested
    if loss_params.get('apply_2d_distortion', True) and 'RDistort' in cam_params and 'TDistort' in cam_params:
        try:
            rd = cam_params['RDistort']
            td = cam_params['TDistort']
            rd_t = torch.as_tensor(np.squeeze(rd), dtype=dtype, device=device)
            td_t = torch.as_tensor(np.squeeze(td), dtype=dtype, device=device)
            distorted = distort_points_torch(proj_2d, K_t, rd_t, td_t)
            return distorted
        except Exception:
            # If distortion fails, return undistorted projection
            pass
    
    return proj_2d



def prepare_batch(batch, device):
    # print(f"🔍 PREPARE_BATCH: Received batch of length {len(batch)}.", flush=True)
    volumes = batch[0].float().to(device)
    grids = batch[1].float().to(device) if batch[1] is not None else None
    targets = batch[2].float().to(device)
    auxs = batch[3].to(device) if batch[3] is not None else None
    keypoints_2d_gt = batch[4] if batch[4] is not None else None
    visibility_2d_gt = batch[6] if len(batch) > 6 and batch[6] is not None else None
    
    # Extract sample IDs for experiment-specific camera parameter matching
    sample_ids = None
    if len(batch) > 5:
        try:
            # Sample IDs might be in various formats, try to extract them
            raw_sample_ids = batch[5]
            if raw_sample_ids is not None:
                # Convert to list of strings if needed
                if hasattr(raw_sample_ids, 'tolist'):
                    sample_ids = [str(sid) for sid in raw_sample_ids.tolist()]
                elif isinstance(raw_sample_ids, (list, tuple)):
                    sample_ids = [str(sid) for sid in raw_sample_ids]
                else:
                    sample_ids = [str(raw_sample_ids)]
        except Exception:
            sample_ids = None
    
    # Build a one-line batch debug string but do not print/log by default
    batch_debug_info = (
        f"🔍 BATCH DEBUG: len={len(batch)}, 2D_labels=None:{batch[4] is None}, "
        f"visibility=None:{visibility_2d_gt is None}, sample_ids={sample_ids[:3] if sample_ids else None}"
    )
    
    # Convert camera-specific 2D data to device
    if keypoints_2d_gt is not None and isinstance(keypoints_2d_gt, dict):
        keypoints_2d_gt_device = {}
        for cam_name, cam_data in keypoints_2d_gt.items():
            if isinstance(cam_data, np.ndarray):
                keypoints_2d_gt_device[cam_name] = torch.from_numpy(cam_data).float().to(device)
            else:
                keypoints_2d_gt_device[cam_name] = cam_data.float().to(device)
        keypoints_2d_gt = keypoints_2d_gt_device

    if visibility_2d_gt is not None and isinstance(visibility_2d_gt, dict):
        visibility_2d_gt_device = {}
        for cam_name, cam_data in visibility_2d_gt.items():
            if isinstance(cam_data, np.ndarray):
                visibility_2d_gt_device[cam_name] = torch.from_numpy(cam_data).bool().to(device)
            else:
                visibility_2d_gt_device[cam_name] = cam_data.bool().to(device)
        visibility_2d_gt = visibility_2d_gt_device
    
    return (
        volumes,
        grids,
        targets,
        auxs,
        keypoints_2d_gt,
        visibility_2d_gt,
        batch_debug_info,
        sample_ids,
    )


def get_visibility_camera_order(visibility_2d):
    if isinstance(visibility_2d, dict):
        return list(visibility_2d.keys())
    return []


def flatten_camera_params(cameras):
    if not isinstance(cameras, dict) or len(cameras) == 0:
        return {}

    sample_val = next(iter(cameras.values()))
    if isinstance(sample_val, dict) and ("K" not in sample_val):
        flattened = {}
        for _exp_idx, cam_map in cameras.items():
            if not isinstance(cam_map, dict):
                continue
            for cam_name, cam_params in cam_map.items():
                flattened[cam_name] = cam_params
        return flattened

    return cameras


def _extract_experiment_id(sample_id):
    if isinstance(sample_id, str) and "_" in sample_id:
        exp_id = sample_id.split("_", 1)[0]
        if exp_id.isdigit():
            return exp_id
    return None


def _resolve_camera_params(flat_cameras, sample_id, cam_name):
    exp_id = _extract_experiment_id(sample_id)
    if exp_id is not None:
        exp_cam_name = f"{exp_id}_{cam_name}"
        if exp_cam_name in flat_cameras:
            return flat_cameras[exp_cam_name]

    if cam_name in flat_cameras:
        return flat_cameras[cam_name]

    for candidate_name, cam_params in flat_cameras.items():
        if candidate_name.endswith(f"_{cam_name}"):
            return cam_params

    return None


def build_visibility_camera_features(
    sample_ids,
    cameras,
    camera_order,
    device=None,
    dtype=torch.float32,
):
    if sample_ids is None or len(camera_order) == 0:
        return None

    flat_cameras = flatten_camera_params(cameras)
    if len(flat_cameras) == 0:
        return None

    features = np.zeros((len(sample_ids), len(camera_order), 6), dtype=np.float32)
    forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for batch_idx, sample_id in enumerate(sample_ids):
        for cam_idx, cam_name in enumerate(camera_order):
            cam_params = _resolve_camera_params(flat_cameras, sample_id, cam_name)
            if cam_params is None:
                continue

            rotation = np.asarray(
                cam_params.get("R", cam_params.get("r")),
                dtype=np.float32,
            ).reshape(3, 3)
            translation = np.asarray(cam_params["t"], dtype=np.float32).reshape(3)
            camera_center = -(rotation.T @ translation)
            camera_forward = rotation.T @ forward_axis
            norm = np.linalg.norm(camera_forward)
            if norm > 1e-6:
                camera_forward = camera_forward / norm

            features[batch_idx, cam_idx, :3] = camera_center
            features[batch_idx, cam_idx, 3:] = camera_forward

    return torch.as_tensor(features, device=device, dtype=dtype)


def visibility_dict_to_tensor(
    visibility_2d, camera_order, device=None, dtype=torch.float32
):
    if visibility_2d is None or len(camera_order) == 0:
        return None

    tensors = []
    ref_shape = None
    for cam_name in camera_order:
        cam_data = visibility_2d.get(cam_name)
        if cam_data is None:
            if ref_shape is None:
                raise ValueError(
                    "Cannot infer visibility tensor shape before seeing a camera entry."
                )
            cam_tensor = torch.zeros(ref_shape, dtype=torch.bool, device=device)
        else:
            if isinstance(cam_data, np.ndarray):
                cam_tensor = torch.from_numpy(cam_data)
            else:
                cam_tensor = cam_data
            if device is not None:
                cam_tensor = cam_tensor.to(device)
            ref_shape = cam_tensor.shape

        if cam_tensor.ndim == 1:
            cam_tensor = cam_tensor.unsqueeze(0)

        if dtype == torch.bool:
            cam_tensor = cam_tensor.bool()
        else:
            cam_tensor = cam_tensor.to(dtype=dtype)

        tensors.append(cam_tensor)

    return torch.stack(tensors, dim=1)


def visibility_tensor_to_dict(visibility_tensor, camera_order):
    if visibility_tensor is None:
        return None
    return {
        cam_name: visibility_tensor[:, cam_idx].bool()
        for cam_idx, cam_name in enumerate(camera_order)
    }


def get_frame_id_from_sample(labels, sample_id, camera_name, exp_idx=None):
    """
    Robust frame ID lookup that handles prefixed camera names.
    
    Args:
        labels: Dataset labels dictionary
        sample_id: Sample ID in format "{experimentID}_{sampleID}"
        camera_name: Camera name (e.g., "Camera1")
        exp_idx: Experiment index (optional, will be parsed from sample_id if not provided)
    
    Returns:
        frame_id: The actual frame index in the video file, or None if not found
    """
    if sample_id not in labels:
        return None
    
    if "frames" not in labels[sample_id]:
        return None
    
    frames_info = labels[sample_id]["frames"]
    
    # Parse experiment index if not provided
    if exp_idx is None and "_" in sample_id:
        exp_idx = int(sample_id.split("_")[0])
    elif exp_idx is None:
        exp_idx = 0
    
    # Try to get the actual frame number for this camera using different naming patterns
    frame_lookup_keys = [
        f"{exp_idx}_{camera_name}",  # "0_Camera1" (most common after prepend_experiment)
        camera_name,                 # "Camera1" (fallback for non-prefixed)
        f"0_{camera_name}",         # "0_Camera1" (explicit experiment 0)
    ]
    
    for key in frame_lookup_keys:
        if key in frames_info:
            frame_id = frames_info[key]
            print(f"  🎬 Found frame number using key '{key}': {frame_id}", flush=True)
            return frame_id
    
    # If no direct match, show available keys for debugging
    print(f"  ❌ Frame lookup failed for camera '{camera_name}' (exp_idx={exp_idx})", flush=True)
    print(f"     Tried keys: {frame_lookup_keys}", flush=True)
    print(f"     Available keys: {list(frames_info.keys())}", flush=True)
    
    # Final fallback: parse from sample_id (but this is likely wrong for multi-experiment scenarios)
    if "_" in sample_id:
        fallback_frame_id = int(sample_id.split("_")[1])
        print(f"  ⚠️ Using fallback frame_id from sample_id: {fallback_frame_id}", flush=True)
        return fallback_frame_id
    
    return None


def get_gt_camera_key(keypoints_2d_gt, camera_name, exp_idx=0):
    """
    Robust GT camera key lookup that handles prefixed camera names.
    
    Args:
        keypoints_2d_gt: Ground truth 2D keypoints dictionary
        camera_name: Camera name (e.g., "Camera1")
        exp_idx: Experiment index
    
    Returns:
        (gt_data, key_used): Tuple of GT data and the key that worked, or (None, None)
    """
    if not isinstance(keypoints_2d_gt, dict):
        return None, None
    
    # Try different ways to match the GT keys with camera names
    possible_gt_keys = [
        f"{exp_idx}_{camera_name}",     # Format: '0_Camera1' (most common after prepend_experiment)
        camera_name,                    # Just camera part: 'Camera1' 
        f"0_{camera_name}",             # Always use 0: '0_Camera1'
        f"{exp_idx}_0_{camera_name}",   # Double prefixed: '0_0_Camera1'
    ]
    
    for gt_key in possible_gt_keys:
        if gt_key in keypoints_2d_gt:
            return keypoints_2d_gt[gt_key], gt_key
    
    # Debug info if no match found
    print(f"  ❌ GT lookup failed for camera '{camera_name}' (exp_idx={exp_idx})", flush=True)
    print(f"     Tried keys: {possible_gt_keys}", flush=True)
    print(f"     Available GT keys: {list(keypoints_2d_gt.keys())}", flush=True)
    
    return None, None


def debug_log_2d_data(batch_idx, epoch, kpts_pred, kpts_pred_2d_dict, keypoints_2d_gt, cameras, kpts_gt=None, checkpoint_dir=None, logger=None):
    """Debug function to log 2D reprojection data for verification"""
    
    # Disabled heavy logger per user request
    return
    
    # Only log first 3 batches
    if batch_idx >= 3:
        return
    
    # Use checkpoint directory if provided, otherwise current directory
    debug_dir = os.path.join(checkpoint_dir or ".", "debug_2d_logs")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create log file for this epoch and batch  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(debug_dir, f"2d_debug_epoch{epoch}_batch{batch_idx}_{timestamp}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"2D Reprojection Debug Log\n")
        f.write(f"========================\n")
        f.write(f"Epoch: {epoch}, Batch: {batch_idx}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        # Log 3D predictions info
        f.write(f"3D Predictions Shape: {kpts_pred.shape}\n")
        f.write(f"3D Predictions Device: {kpts_pred.device}\n")
        f.write(f"3D Predictions Sample (batch 0, first 3 keypoints):\n")
        kpts_3d_sample = kpts_pred[0, :, :3].detach().cpu().numpy()
        for i in range(min(3, kpts_3d_sample.shape[1])):
            f.write(f"  Keypoint {i}: [{kpts_3d_sample[0,i]:.3f}, {kpts_3d_sample[1,i]:.3f}, {kpts_3d_sample[2,i]:.3f}]\n")
        
        # Add comprehensive 3D predictions analysis
        f.write(f"\nDetailed 3D Predictions Analysis:\n")
        kpts_3d_full = kpts_pred.detach().cpu().numpy()  # [B, 3, n_keypoints]
        f.write(f"  Batch size: {kpts_3d_full.shape[0]}\n")
        f.write(f"  Dimensions: {kpts_3d_full.shape[1]}\n") 
        f.write(f"  Keypoints: {kpts_3d_full.shape[2]}\n")
        
        for b in range(min(2, kpts_3d_full.shape[0])):  # Show first 2 batches
            f.write(f"\nBatch {b} 3D Predictions:\n")
            batch_3d = kpts_3d_full[b]  # [3, n_keypoints]
            
            # Statistics
            f.write(f"  X range: [{batch_3d[0].min():.3f}, {batch_3d[0].max():.3f}]\n")
            f.write(f"  Y range: [{batch_3d[1].min():.3f}, {batch_3d[1].max():.3f}]\n")
            f.write(f"  Z range: [{batch_3d[2].min():.3f}, {batch_3d[2].max():.3f}]\n")
            
            # Individual keypoints
            f.write(f"  All keypoints (x, y, z):\n")
            for i in range(batch_3d.shape[1]):
                x, y, z = batch_3d[0, i], batch_3d[1, i], batch_3d[2, i]
                f.write(f"    Kpt{i:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})\n")
        
        # Log 3D ground truth info if provided
        if kpts_gt is not None:
            f.write(f"\n3D Ground Truth Analysis:\n")
            if isinstance(kpts_gt, torch.Tensor):
                kpts_gt_np = kpts_gt.detach().cpu().numpy()
            else:
                kpts_gt_np = np.array(kpts_gt)
            
            f.write(f"  GT Shape: {kpts_gt_np.shape}\n")
            f.write(f"  GT Sample (batch 0, first 3 keypoints):\n")
            
            gt_sample = kpts_gt_np[0] if len(kpts_gt_np.shape) > 2 else kpts_gt_np
            for i in range(min(3, gt_sample.shape[1] if len(gt_sample.shape) > 1 else 1)):
                if len(gt_sample.shape) > 1:
                    f.write(f"    GT Keypoint {i}: [{gt_sample[0,i]:.3f}, {gt_sample[1,i]:.3f}, {gt_sample[2,i]:.3f}]\n")
            
            # Full GT analysis
            for b in range(min(2, kpts_gt_np.shape[0] if len(kpts_gt_np.shape) > 2 else 1)):
                f.write(f"\nBatch {b} 3D Ground Truth:\n")
                if len(kpts_gt_np.shape) > 2:
                    batch_gt = kpts_gt_np[b]  # [3, n_keypoints]
                else:
                    batch_gt = kpts_gt_np
                
                # Check for NaN values
                nan_count = np.isnan(batch_gt).sum()
                total_elements = batch_gt.size
                f.write(f"  NaN values: {nan_count}/{total_elements} ({100*nan_count/total_elements:.1f}%)\n")
                
                if nan_count < total_elements:  # If not all NaN
                    valid_data = batch_gt[~np.isnan(batch_gt)]
                    if len(valid_data) > 0:
                        f.write(f"  Valid data range: [{valid_data.min():.3f}, {valid_data.max():.3f}]\n")
                
                # Individual GT keypoints
                f.write(f"  All GT keypoints (x, y, z):\n")
                for i in range(batch_gt.shape[1] if len(batch_gt.shape) > 1 else 1):
                    if len(batch_gt.shape) > 1:
                        x, y, z = batch_gt[0, i], batch_gt[1, i], batch_gt[2, i]
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            f.write(f"    GT Kpt{i:2d}: NaN values\n")
                        else:
                            f.write(f"    GT Kpt{i:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})\n")
        else:
            f.write(f"\n3D Ground Truth: Not provided\n")
        
        f.write("\n")
        
        # Log camera information
        f.write(f"Available Cameras: {list(cameras.keys())}\n")
        for cam_name, cam_params in cameras.items():
            f.write(f"\nCamera: {cam_name}\n")
            f.write(f"  K matrix shape: {cam_params['K'].shape}\n")
            f.write(f"  K matrix:\n{cam_params['K']}\n")
            R = cam_params.get('R', cam_params.get('r'))
            f.write(f"  R matrix shape: {R.shape}\n")
            f.write(f"  R matrix:\n{R}\n")
            f.write(f"  t vector shape: {cam_params['t'].shape}\n")
            f.write(f"  t vector: {cam_params['t']}\n")
        f.write("\n")
        
        # Log ground truth 2D data
        if keypoints_2d_gt is not None:
            f.write(f"Ground Truth 2D Data:\n")
            f.write(f"Type: {type(keypoints_2d_gt)}\n")
            
            if isinstance(keypoints_2d_gt, dict):
                for cam_name, gt_2d in keypoints_2d_gt.items():
                    f.write(f"\nCamera {cam_name} Ground Truth:\n")
                    f.write(f"  Shape: {gt_2d.shape}\n")
                    f.write(f"  Device: {gt_2d.device}\n")
                    
                    # Sample data for first batch, first 3 keypoints
                    gt_sample = gt_2d[0, :, :3].detach().cpu().numpy()
                    f.write(f"  Sample (batch 0, first 3 keypoints):\n")
                    for i in range(min(3, gt_sample.shape[1])):
                        f.write(f"    Keypoint {i}: x={gt_sample[0,i]:.3f}, y={gt_sample[1,i]:.3f}\n")
                    
                    # Check for NaN values
                    nan_count = torch.isnan(gt_2d).sum().item()
                    total_elements = gt_2d.numel()
                    f.write(f"  NaN values: {nan_count}/{total_elements} ({100*nan_count/total_elements:.1f}%)\n")
            else:
                f.write(f"  Shape: {keypoints_2d_gt.shape}\n")
                f.write(f"  Device: {keypoints_2d_gt.device}\n")
        else:
            f.write("Ground Truth 2D Data: None\n")
        f.write("\n")
        
        # Log projected 2D data
        if kpts_pred_2d_dict is not None:
            f.write(f"Projected 2D Data:\n")
            for cam_name, pred_2d in kpts_pred_2d_dict.items():
                f.write(f"\nCamera {cam_name} Projections:\n")
                f.write(f"  Shape: {pred_2d.shape}\n")
                f.write(f"  Device: {pred_2d.device}\n")
                
                # Sample data for first batch, first 3 keypoints
                pred_sample = pred_2d[0, :, :3].detach().cpu().numpy()
                f.write(f"  Sample (batch 0, first 3 keypoints):\n")
                for i in range(min(3, pred_sample.shape[1])):
                    f.write(f"    Keypoint {i}: x={pred_sample[0,i]:.3f}, y={pred_sample[1,i]:.3f}\n")
                
                # Check for invalid values
                nan_count = torch.isnan(pred_2d).sum().item()
                inf_count = torch.isinf(pred_2d).sum().item()
                total_elements = pred_2d.numel()
                f.write(f"  NaN values: {nan_count}/{total_elements}\n")
                f.write(f"  Inf values: {inf_count}/{total_elements}\n")
        f.write("\n")
        
        # Compare projections with ground truth
        if kpts_pred_2d_dict is not None and keypoints_2d_gt is not None and isinstance(keypoints_2d_gt, dict):
            f.write(f"Projection vs Ground Truth Comparison:\n")
            for cam_name in cameras.keys():
                if cam_name in kpts_pred_2d_dict and cam_name in keypoints_2d_gt:
                    pred_2d = kpts_pred_2d_dict[cam_name]
                    gt_2d = keypoints_2d_gt[cam_name]
                    
                    f.write(f"\nCamera {cam_name}:\n")
                    
                    # Calculate differences for first batch, up to the number of available GT keypoints
                    pred_sample = pred_2d[0, :, :3].detach().cpu().numpy()
                    gt_sample = gt_2d[0, :, :3].detach().cpu().numpy()
                    common_k = min(pred_sample.shape[1], gt_sample.shape[1])
                    for i in range(common_k):
                        if not (np.isnan(gt_sample[0, i]) or np.isnan(gt_sample[1, i])):
                            dx = pred_sample[0,i] - gt_sample[0,i]
                            dy = pred_sample[1,i] - gt_sample[1,i]
                            dist = np.sqrt(dx*dx + dy*dy)
                            f.write(f"  Keypoint {i}:\n")
                            f.write(f"    Predicted: ({pred_sample[0,i]:.3f}, {pred_sample[1,i]:.3f})\n")
                            f.write(f"    Ground Truth: ({gt_sample[0,i]:.3f}, {gt_sample[1,i]:.3f})\n")
                            f.write(f"    Difference: ({dx:.3f}, {dy:.3f})\n")
                            f.write(f"    Distance: {dist:.3f} pixels\n")
                        else:
                            f.write(f"  Keypoint {i}: Ground Truth has NaN values\n")
    
    # Force immediate file write
    with open(log_file, 'w') as f:
        pass  # File is already written above, this just ensures it's flushed
    
    # Create immediate console output and log it
    debug_summary = f"\n{'='*60}\n2D REPROJECTION DEBUG - Epoch {epoch}, Batch {batch_idx}\n{'='*60}"
    
    if logger:
        logger.info(f"Debug log saved to: {log_file}")
        logger.info(debug_summary)
    else:
        print(f"Debug log saved to: {log_file}", flush=True)
        print(debug_summary, flush=True)
    
    # Print comprehensive 3D predictions analysis
    kpts_3d_full = kpts_pred.detach().cpu().numpy()  # [B, 3, n_keypoints]
    print(f"\n🎯 3D PREDICTIONS ANALYSIS", flush=True)
    print(f"Shape: {kpts_pred.shape} (Batch, XYZ, Keypoints)", flush=True)
    print(f"Device: {kpts_pred.device}", flush=True)
    
    for b in range(min(2, kpts_3d_full.shape[0])):  # Show first 2 batches
        print(f"\n📊 Batch {b} 3D Predictions:", flush=True)
        batch_3d = kpts_3d_full[b]  # [3, n_keypoints]
        
        # Statistics
        print(f"  📐 Coordinate ranges:", flush=True)
        print(f"    X: [{batch_3d[0].min():8.3f}, {batch_3d[0].max():8.3f}]", flush=True)
        print(f"    Y: [{batch_3d[1].min():8.3f}, {batch_3d[1].max():8.3f}]", flush=True) 
        print(f"    Z: [{batch_3d[2].min():8.3f}, {batch_3d[2].max():8.3f}]", flush=True)
        
        # Center of mass
        center_x = batch_3d[0].mean()
        center_y = batch_3d[1].mean()
        center_z = batch_3d[2].mean()
        print(f"  🎯 Center of mass: ({center_x:8.3f}, {center_y:8.3f}, {center_z:8.3f})", flush=True)
        
        # Individual keypoints (show all)
        print(f"  📍 All keypoints (x, y, z):", flush=True)
        for i in range(batch_3d.shape[1]):
            x, y, z = batch_3d[0, i], batch_3d[1, i], batch_3d[2, i]
            print(f"    Kpt{i:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})", flush=True)
    
    # Print 3D ground truth analysis if provided
    if kpts_gt is not None:
        print(f"\n🎯 3D GROUND TRUTH ANALYSIS", flush=True)
        if isinstance(kpts_gt, torch.Tensor):
            kpts_gt_np = kpts_gt.detach().cpu().numpy()
        else:
            kpts_gt_np = np.array(kpts_gt)
        
        print(f"GT Shape: {kpts_gt_np.shape}", flush=True)
        
        for b in range(min(2, kpts_gt_np.shape[0] if len(kpts_gt_np.shape) > 2 else 1)):
            print(f"\n📊 Batch {b} 3D Ground Truth:", flush=True)
            if len(kpts_gt_np.shape) > 2:
                batch_gt = kpts_gt_np[b]  # [3, n_keypoints]
            else:
                batch_gt = kpts_gt_np
            
            # Check for NaN values
            nan_count = np.isnan(batch_gt).sum()
            total_elements = batch_gt.size
            print(f"  🔍 NaN values: {nan_count}/{total_elements} ({100*nan_count/total_elements:.1f}%)", flush=True)
            
            if nan_count < total_elements:  # If not all NaN
                valid_data = batch_gt[~np.isnan(batch_gt)]
                if len(valid_data) > 0:
                    print(f"  📐 Valid data range: [{valid_data.min():.3f}, {valid_data.max():.3f}]", flush=True)
                    
                    # Show valid coordinate ranges if we have valid data
                    if len(batch_gt.shape) > 1 and batch_gt.shape[0] >= 3:
                        x_valid = batch_gt[0][~np.isnan(batch_gt[0])]
                        y_valid = batch_gt[1][~np.isnan(batch_gt[1])]
                        z_valid = batch_gt[2][~np.isnan(batch_gt[2])]
                        if len(x_valid) > 0:
                            print(f"    X: [{x_valid.min():8.3f}, {x_valid.max():8.3f}]", flush=True)
                        if len(y_valid) > 0:
                            print(f"    Y: [{y_valid.min():8.3f}, {y_valid.max():8.3f}]", flush=True)
                        if len(z_valid) > 0:
                            print(f"    Z: [{z_valid.min():8.3f}, {z_valid.max():8.3f}]", flush=True)
            
            # Individual GT keypoints
            print(f"  📍 All GT keypoints (x, y, z):", flush=True)
            for i in range(batch_gt.shape[1] if len(batch_gt.shape) > 1 else 1):
                if len(batch_gt.shape) > 1:
                    x, y, z = batch_gt[0, i], batch_gt[1, i], batch_gt[2, i]
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        print(f"    GT Kpt{i:2d}: NaN values", flush=True)
                    else:
                        print(f"    GT Kpt{i:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})", flush=True)
    else:
        print(f"\n❌ 3D Ground Truth: Not provided to debug function", flush=True)
    
    print(f"{'='*60}", flush=True)
    
    # Print camera info
    print(f"\nAvailable Cameras: {list(cameras.keys())}", flush=True)
    
    # Print ground truth 2D info
    if keypoints_2d_gt is not None and isinstance(keypoints_2d_gt, dict):
        print(f"\nGround Truth 2D Data:")
        for cam_name, gt_2d in keypoints_2d_gt.items():
            gt_sample = gt_2d[0, :, :3].detach().cpu().numpy()
            nan_count = torch.isnan(gt_2d).sum().item()
            total_elements = gt_2d.numel()
            print(f"  {cam_name} Shape: {gt_2d.shape}, NaN: {nan_count}/{total_elements}")
            # Print up to first 2 available keypoints safely
            max_show = min(2, gt_sample.shape[1])
            for i in range(max_show):
                x, y = gt_sample[0, i], gt_sample[1, i]
                if np.isnan(x) or np.isnan(y):
                    print(f"    Kpt{i}=NaN", flush=True)
                else:
                    print(f"    Kpt{i}=({x:.1f},{y:.1f})", flush=True)
    
    # Print projected 2D info and comparisons
    if kpts_pred_2d_dict is not None:
        print(f"\nProjected 2D Data & Comparisons:")
        for cam_name, pred_2d in kpts_pred_2d_dict.items():
            pred_sample = pred_2d[0, :, :3].detach().cpu().numpy()
            print(f"  {cam_name} Projections Shape: {pred_2d.shape}", flush=True)
            # Print up to first 2 predicted points safely
            max_show_pred = min(2, pred_sample.shape[1])
            for i in range(max_show_pred):
                print(f"    Kpt{i}=({pred_sample[0,i]:.1f},{pred_sample[1,i]:.1f})", flush=True)
            
            # Show comparison if ground truth available
            if keypoints_2d_gt is not None and isinstance(keypoints_2d_gt, dict) and cam_name in keypoints_2d_gt:
                gt_2d = keypoints_2d_gt[cam_name]
                gt_sample = gt_2d[0, :, :3].detach().cpu().numpy()
                
                print(f"    Pixel Errors:")
                for i in range(min(2, pred_sample.shape[1])):  # Show first 2 keypoints
                    if not (np.isnan(gt_sample[0,i]) or np.isnan(gt_sample[1,i])):
                        dx = pred_sample[0,i] - gt_sample[0,i]
                        dy = pred_sample[1,i] - gt_sample[1,i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        print(f"      Kpt{i}: Pred({pred_sample[0,i]:.1f},{pred_sample[1,i]:.1f}) vs GT({gt_sample[0,i]:.1f},{gt_sample[1,i]:.1f}) -> Error: {dist:.1f}px", flush=True)
                    else:
                        print(f"      Kpt{i}: Ground Truth has NaN values", flush=True)
    
    print(f"{'='*60}\n", flush=True)



class LossHelper:
    def __init__(self, params, checkpoint_dir=None, logger=None):
        self.loss_params = params
        self.debug_batch_counter = 0  # Track batches for debug logging
        self.current_epoch = 0  # Track current epoch
        self.checkpoint_dir = checkpoint_dir  # For saving debug logs
        self.logger = logger  # For logging messages
        self._get_losses()

    def _get_losses(self):
        # Backward-compatible: build standard loss map
        self.loss_fcns = {}
        if "loss" in self.loss_params:
            for name, args in self.loss_params["loss"].items():
                name_clean = str(name).strip()
                if name_clean.endswith(":"):
                    name_clean = name_clean[:-1].strip()
                self.loss_fcns[name_clean] = getattr(custom_losses, name_clean)(**args)

        # New: allow explicitly specifying separate sets for 3D and 2D
        # If not provided, fall back to the standard map above
        self.loss_fcns_3d = {}
        if "loss_3d" in self.loss_params:
            for name, args in self.loss_params["loss_3d"].items():
                name_clean = str(name).strip()
                if name_clean.endswith(":"):
                    name_clean = name_clean[:-1].strip()
                self.loss_fcns_3d[name_clean] = getattr(custom_losses, name_clean)(**args)
        else:
            # default to previously parsed losses
            self.loss_fcns_3d = dict(self.loss_fcns)

        self.loss_fcns_2d = {}
        if "loss_2d" in self.loss_params:
            for name, args in self.loss_params["loss_2d"].items():
                name_clean = str(name).strip()
                if name_clean.endswith(":"):
                    name_clean = name_clean[:-1].strip()
                
                # Handle both dictionary and string format arguments
                if isinstance(args, str):
                    # Parse string format like 'loss_weight:0.4' into dict
                    parsed_args = {}
                    for param_str in args.split(','):
                        if ':' in param_str:
                            key, value = param_str.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            # Try to convert to appropriate type
                            try:
                                if '.' in value:
                                    parsed_args[key] = float(value)
                                else:
                                    parsed_args[key] = int(value)
                            except ValueError:
                                parsed_args[key] = value  # Keep as string if conversion fails
                    args = parsed_args
                
                self.loss_fcns_2d[name_clean] = getattr(custom_losses, name_clean)(**args)
    
    def set_epoch(self, epoch):
        """Set current epoch for debug logging"""
        self.current_epoch = epoch
        self.debug_batch_counter = 0  # Reset batch counter for new epoch

    def compute_confidence_weights(self, confidence_scores, method="linear", strength=1.0):
        """Compute confidence weights for loss weighting.
        
        Args:
            confidence_scores (torch.Tensor): Raw confidence scores [0,1]
            method (str): Weighting method - 'linear', 'sigmoid', 'exponential'
            strength (float): Scaling factor for confidence influence
            
        Returns:
            torch.Tensor: Confidence weights for loss computation
        """
        if method == "linear":
            # Linear mapping: confidence directly becomes weight
            weights = confidence_scores * strength
        elif method == "sigmoid":
            # Sigmoid mapping: enhances contrast between high/low confidence
            # Maps [0,1] -> approximately [0.05, 0.95] with steeper transition
            scaled_conf = (confidence_scores - 0.5) * 6 * strength
            weights = torch.sigmoid(scaled_conf)
        elif method == "exponential":
            # Exponential mapping: gives very high weight to high confidence
            weights = torch.pow(confidence_scores, 1.0 / strength)
        else:
            # Fallback to linear
            weights = confidence_scores * strength
            
        return weights.clamp(min=0.01, max=10.0)  # Prevent extreme weights

    def apply_confidence_weighting(self, loss_tensor, confidence_tensor, method="linear", strength=1.0):
        """Apply confidence weighting to loss tensor.
        
        Args:
            loss_tensor (torch.Tensor): Computed loss values
            confidence_tensor (torch.Tensor): Confidence scores matching loss shape
            method (str): Confidence weighting method
            strength (float): Confidence weighting strength
            
        Returns:
            torch.Tensor: Confidence-weighted loss
        """
        if confidence_tensor is None:
            return loss_tensor
            
        # Ensure tensors are on same device
        if confidence_tensor.device != loss_tensor.device:
            confidence_tensor = confidence_tensor.to(loss_tensor.device)
            
        # Compute confidence weights
        conf_weights = self.compute_confidence_weights(confidence_tensor, method, strength)
        
        # Apply weighting
        weighted_loss = loss_tensor * conf_weights
        
        return weighted_loss

    def compute_loss(
        self,
        kpts_gt,
        kpts_pred,
        heatmaps,
        grid_centers=None,
        aux=None,
        heatmaps_gt=None,
        keypoints_2d_gt=None,
        visibility_2d_gt=None,
        cameras=None,
        sample_ids=None,  # Add sample_ids to determine experiment for each sample
        confidence_2d_gt=None,  # Add 2D confidence data for weighting
    ):
        """
        Compute each loss and return their weighted sum for backprop.
        """
        loss_dict = {}
        total_loss = []

        # The 3D loss is always computed. Missing labels are handled by NaN masking in the loss function
        # Use explicitly configured 3D losses if provided; otherwise fall back to legacy map
        for k, lossfcn in self.loss_fcns_3d.items():
            if k == "GaussianRegLoss":
                loss_val = lossfcn(
                    kpts_gt,
                    kpts_pred.clone().detach(),
                    heatmaps,
                    grid_centers.clone().detach(),
                )
            elif k == "MSELoss" or k == "BCELoss":
                if heatmaps_gt is not None:
                    loss_val = lossfcn(heatmaps_gt, heatmaps)
                else:
                    loss_val = lossfcn(kpts_gt, heatmaps)
            elif "SilhouetteLoss" in k or k == "ReconstructionLoss":
                loss_val = lossfcn(aux, heatmaps)
            elif k == "VarianceLoss":
                loss_val = lossfcn(kpts_pred, heatmaps, grid_centers)
            else:
                loss_val = lossfcn(kpts_gt, kpts_pred)
            total_loss.append(loss_val)
            loss_dict[k] = loss_val.detach().clone().cpu().item()

        # DEBUG: Always print and log for ALL batches
        # print("\n" + "="*80, flush=True)  # DISABLED
        # print(f"🔍 LOSS COMPUTATION BATCH {self.debug_batch_counter} - EPOCH {self.current_epoch}", flush=True)  # DISABLED
        # print(f"2D data is None: {keypoints_2d_gt is None}", flush=True)  # DISABLED
        # print(f"train_on_2d setting: {self.loss_params.get('train_on_2d', 'NOT_FOUND')}", flush=True)  # DISABLED
        if keypoints_2d_gt is not None:
            # print(f"2D data type: {type(keypoints_2d_gt)}", flush=True)  # DISABLED
            if isinstance(keypoints_2d_gt, dict):
                # print(f"2D data cameras: {list(keypoints_2d_gt.keys())}", flush=True)  # DISABLED
                # for k, v in keypoints_2d_gt.items():
                #     print(f"  {k}: {getattr(v, 'shape', 'NO_SHAPE')}", flush=True)  # DISABLED
                pass
        # print("="*80, flush=True)  # DISABLED
        
        # Reduced: no file debug per request
        
        # If 2D labels are available, compute the 2D reprojection loss
        if keypoints_2d_gt is not None and self.loss_params.get("train_on_2d", True):
            # print("🎯 ENTERING 2D LOSS COMPUTATION!", flush=True)  # DISABLED
            
            # Handle cameras dict: maintain experiment-specific parameters for accurate 2D reprojection
            # Do NOT merge cameras by base name - each experiment may have different calibrations!
            flat_cameras = flatten_camera_params(cameras)

            # Debug: inspect camera dict before projection
            # try:
            #     cam_keys = list(flat_cameras.keys()) if isinstance(flat_cameras, dict) else []
            #     print(f"📷 Cameras available for 2D projection: {cam_keys[:5]}{'...' if len(cam_keys)>5 else ''}")
            #     if cam_keys:
            #         sample_cam = cam_keys[0]
            #         sample_params = flat_cameras[sample_cam]
            #         print(f"  Sample cam: {sample_cam}, keys: {list(sample_params.keys())}")
            #         for need in ['K','R','r','t']:
            #             present = need in sample_params
            #             print(f"   - has {need}: {present}")
            # except Exception as _e:
            #     print(f"(Camera debug skipped: {_e})")  # DISABLED

            # Project the 3D predictions to 2D using experiment-aware camera parameters
            kpts_pred_2d_dict = {}
            batch_size, _, n_keypoints = kpts_pred.shape
            
            # Debug information about the batch
            if isinstance(keypoints_2d_gt, dict) and self.debug_batch_counter < 3:
                # Determine which experiments are present in this batch
                batch_experiments = set()
                if sample_ids is not None:
                    for sample_id in sample_ids:
                        if isinstance(sample_id, str) and '_' in sample_id:
                            exp_id = sample_id.split('_')[0]
                            if exp_id.isdigit():
                                batch_experiments.add(exp_id)
                
                available_cam_params = list(flat_cameras.keys())
                print(f"🎯 [2D LOSS] Batch contains experiments: {sorted(batch_experiments) if batch_experiments else 'unknown'}", flush=True)
                if len(batch_experiments) > 1:
                    print(f"✅ [2D LOSS] Mixed-experiment batch detected. Using experiment-aware projection.", flush=True)
                print(f"🎯 [2D LOSS] Available camera parameters: {len(available_cam_params)} cameras", flush=True)
                print(f"🎯 [2D LOSS] GT cameras needing parameters: {list(keypoints_2d_gt.keys())}", flush=True)
            
            # Prepare debug info for the projection function
            debug_info = {
                'debug_batch_counter': self.debug_batch_counter
            }
            
            # Get the list of cameras that have GT data
            gt_camera_names = []
            if isinstance(keypoints_2d_gt, dict):
                gt_camera_names = list(keypoints_2d_gt.keys())
            else:
                # If GT is not a dict, project for all available cameras
                # Extract unique base camera names from flat_cameras
                base_cams = set()
                for cam_name in flat_cameras.keys():
                    if '_' in cam_name and cam_name.split('_')[0].isdigit():
                        base_cam = '_'.join(cam_name.split('_')[1:])
                        base_cams.add(base_cam)
                    else:
                        base_cams.add(cam_name)
                gt_camera_names = list(base_cams)
            
            # For each GT camera, use experiment-aware projection
            for cam_name in gt_camera_names:
                # Use the new experiment-aware projection function
                proj_2d = project_batch_with_experiment_aware_cameras(
                    kpts_pred=kpts_pred,
                    sample_ids=sample_ids,
                    flat_cameras=flat_cameras,
                    cam_name=cam_name,
                    loss_params=self.loss_params,
                    debug_info=debug_info
                )
                
                # Store the projection results
                kpts_pred_2d_dict[cam_name] = proj_2d  # [B, 2, N], requires_grad=True
            
            # Debug logging for first 3 batches
            debug_log_2d_data(
                batch_idx=self.debug_batch_counter,
                epoch=self.current_epoch,
                kpts_pred=kpts_pred,
                kpts_pred_2d_dict=kpts_pred_2d_dict,
                keypoints_2d_gt=keypoints_2d_gt,
                cameras=flat_cameras,  # Pass all cameras
                kpts_gt=kpts_gt,  # Pass 3D ground truth
                checkpoint_dir=self.checkpoint_dir,
                logger=self.logger
            )
            self.debug_batch_counter += 1
        else:
            # print("❌ NOT ENTERING 2D LOSS COMPUTATION", flush=True)  # DISABLED
            
            # Reduced: no file debug per request
            
            self.debug_batch_counter += 1  # Increment counter even if not entering 2D loss
            
        # Continue with the rest of the 2D loss computation if we entered the 2D block
        if keypoints_2d_gt is not None and self.loss_params.get("train_on_2d", False):
            # Preferred path: use explicitly configured 2D losses if provided
            selected_2d_losses = self.loss_fcns_2d if getattr(self, "loss_fcns_2d", None) else {}

            # Backward-compatible fallback: choose the best available 2D-suitable loss
            if not selected_2d_losses:
                candidates = [
                    name for name in ["CharbonnierLoss", "HuberLoss", "L1Loss", "L2Loss", "WeightedL1Loss"]
                    if name in self.loss_fcns
                ]
                if len(candidates) > 0:
                    best_name = max(candidates, key=lambda n: getattr(self.loss_fcns[n], "loss_weight", 0.0))
                    if getattr(self.loss_fcns[best_name], "loss_weight", 0.0) > 0.0:
                        selected_2d_losses = {best_name: self.loss_fcns[best_name]}
                    else:
                        selected_2d_losses = {}
                else:
                    selected_2d_losses = {}

            # Reduced: no file debug per request

            # Compute configured 2D losses (each aggregated across cameras)
            for loss_name, loss_fcn in selected_2d_losses.items():
                total_2d_loss = 0
                n_cameras = 0

                # Compute only over cameras that have GT for this sample; skip cameras with all-NaN GT
                used_cameras = []
                common_cameras = sorted(set(kpts_pred_2d_dict.keys()) & set(keypoints_2d_gt.keys()))
                
                for cam_name in common_cameras:
                    kpts_pred_2d_cam = kpts_pred_2d_dict[cam_name]
                    keypoints_2d_gt_cam = keypoints_2d_gt[cam_name]

                    if not isinstance(keypoints_2d_gt_cam, torch.Tensor):
                        keypoints_2d_gt_cam = torch.from_numpy(keypoints_2d_gt_cam).float().to(kpts_pred.device)
                    else:
                        keypoints_2d_gt_cam = keypoints_2d_gt_cam.to(kpts_pred.device)

                    # Ensure GT and predictions have matching batch dimensions
                    pred_batch_size = kpts_pred_2d_cam.shape[0]
                    gt_batch_size = keypoints_2d_gt_cam.shape[0]
                    
                    if gt_batch_size != pred_batch_size:
                        # If GT has fewer samples, pad with NaN to match prediction batch size
                        if gt_batch_size < pred_batch_size:
                            pad_size = pred_batch_size - gt_batch_size
                            nan_pad = torch.full(
                                (pad_size, keypoints_2d_gt_cam.shape[1], keypoints_2d_gt_cam.shape[2]),
                                float('nan'),
                                device=keypoints_2d_gt_cam.device,
                                dtype=keypoints_2d_gt_cam.dtype
                            )
                            keypoints_2d_gt_cam = torch.cat([keypoints_2d_gt_cam, nan_pad], dim=0)
                        else:
                            # If GT has more samples, truncate to match prediction batch size
                            keypoints_2d_gt_cam = keypoints_2d_gt_cam[:pred_batch_size]

                    if (
                        self.loss_params.get("exclude_occluded_2d", False)
                        and visibility_2d_gt is not None
                        and cam_name in visibility_2d_gt
                    ):
                        visibility_2d_cam = visibility_2d_gt[cam_name]
                        if not isinstance(visibility_2d_cam, torch.Tensor):
                            visibility_2d_cam = torch.from_numpy(visibility_2d_cam).bool().to(kpts_pred.device)
                        else:
                            visibility_2d_cam = visibility_2d_cam.to(kpts_pred.device).bool()

                        if visibility_2d_cam.shape[0] != pred_batch_size:
                            if visibility_2d_cam.shape[0] < pred_batch_size:
                                pad_size = pred_batch_size - visibility_2d_cam.shape[0]
                                false_pad = torch.zeros(
                                    (pad_size, visibility_2d_cam.shape[1]),
                                    device=visibility_2d_cam.device,
                                    dtype=torch.bool,
                                )
                                visibility_2d_cam = torch.cat([visibility_2d_cam, false_pad], dim=0)
                            else:
                                visibility_2d_cam = visibility_2d_cam[:pred_batch_size]

                        invisible_mask = (~visibility_2d_cam).unsqueeze(1).expand_as(
                            keypoints_2d_gt_cam
                        )
                        keypoints_2d_gt_cam = keypoints_2d_gt_cam.clone()
                        keypoints_2d_gt_cam = keypoints_2d_gt_cam.masked_fill(
                            invisible_mask, float("nan")
                        )

                    # Skip cameras with no valid GT points
                    valid_gt_mask = ~torch.isnan(keypoints_2d_gt_cam)
                    n_valid_gt = valid_gt_mask.sum().item()
                    if n_valid_gt == 0:
                        continue

                    # Compute base 2D loss
                    loss_val_cam = loss_fcn(keypoints_2d_gt_cam, kpts_pred_2d_cam)
                    
                    # Apply confidence weighting if available and enabled
                    if (confidence_2d_gt is not None and 
                        self.loss_params.get("use_2d_confidence_weighting", False) and
                        cam_name in confidence_2d_gt):
                        
                        confidence_2d_cam = confidence_2d_gt[cam_name]
                        if not isinstance(confidence_2d_cam, torch.Tensor):
                            confidence_2d_cam = torch.from_numpy(confidence_2d_cam).float().to(kpts_pred.device)
                        else:
                            confidence_2d_cam = confidence_2d_cam.to(kpts_pred.device)
                        
                        # Ensure confidence tensor matches the loss computation dimensions
                        if confidence_2d_cam.shape[0] != pred_batch_size:
                            if confidence_2d_cam.shape[0] < pred_batch_size:
                                # Pad confidence with default values for missing samples
                                pad_size = pred_batch_size - confidence_2d_cam.shape[0]
                                default_conf = torch.ones((pad_size, confidence_2d_cam.shape[1]), 
                                                         device=confidence_2d_cam.device,
                                                         dtype=confidence_2d_cam.dtype)
                                confidence_2d_cam = torch.cat([confidence_2d_cam, default_conf], dim=0)
                            else:
                                confidence_2d_cam = confidence_2d_cam[:pred_batch_size]
                        
                        # Apply threshold filtering - set low confidence points to very low weight
                        min_threshold = self.loss_params.get("min_confidence_threshold", 0.5)
                        confidence_2d_cam = torch.where(
                            confidence_2d_cam < min_threshold, 
                            torch.tensor(0.01, device=confidence_2d_cam.device), 
                            confidence_2d_cam
                        )
                        
                        # Compute confidence weights based on method
                        weighting_method = self.loss_params.get("confidence_loss_weighting_method", "linear")
                        weighting_strength = self.loss_params.get("confidence_weighting_strength", 1.0)
                        
                        # For loss functions that return per-sample losses, apply per-point weighting
                        if hasattr(loss_fcn, 'reduction') and loss_fcn.reduction == 'none':
                            # Expand confidence to match loss tensor dimensions if needed
                            if confidence_2d_cam.dim() == 2 and loss_val_cam.dim() == 3:
                                # Expand from (batch, keypoints) to (batch, keypoints, 2) for x,y coordinates
                                confidence_expanded = confidence_2d_cam.unsqueeze(-1).expand_as(loss_val_cam)
                            else:
                                confidence_expanded = confidence_2d_cam
                            
                            # Apply confidence weighting to per-point losses
                            loss_val_cam = self.apply_confidence_weighting(
                                loss_val_cam, confidence_expanded, weighting_method, weighting_strength
                            )
                            # Reduce weighted losses
                            loss_val_cam = loss_val_cam.mean()
                        else:
                            # For aggregated losses, apply mean confidence weighting
                            mean_confidence = confidence_2d_cam[~torch.isnan(keypoints_2d_gt_cam[:,:,0])].mean()
                            conf_weight = self.compute_confidence_weights(
                                mean_confidence.unsqueeze(0), weighting_method, weighting_strength
                            ).item()
                            loss_val_cam = loss_val_cam * conf_weight
                    
                    # Debug high loss values
                    if loss_val_cam.item() > 70 and self.debug_batch_counter < 10:
                        print(f"\n⚠️  HIGH 2D LOSS DETECTED: {loss_val_cam.item():.4f} for camera {cam_name}", flush=True)
                        print(f"   Batch samples: {sample_ids[:5] if sample_ids else 'Unknown'}", flush=True)
                        
                        # Check for extreme projection values
                        pred_max = kpts_pred_2d_cam.max().item()
                        pred_min = kpts_pred_2d_cam.min().item()
                        gt_max = keypoints_2d_gt_cam[~torch.isnan(keypoints_2d_gt_cam)].max().item() if (~torch.isnan(keypoints_2d_gt_cam)).any() else float('nan')
                        gt_min = keypoints_2d_gt_cam[~torch.isnan(keypoints_2d_gt_cam)].min().item() if (~torch.isnan(keypoints_2d_gt_cam)).any() else float('nan')
                        
                        print(f"   Pred range: [{pred_min:.2f}, {pred_max:.2f}]", flush=True)
                        print(f"   GT range: [{gt_min:.2f}, {gt_max:.2f}]", flush=True)
                        
                        # Check for NaN/Inf in predictions
                        nan_pred = torch.isnan(kpts_pred_2d_cam).sum().item()
                        inf_pred = torch.isinf(kpts_pred_2d_cam).sum().item()
                        if nan_pred > 0 or inf_pred > 0:
                            print(f"   ⚠️  Pred contains {nan_pred} NaN and {inf_pred} Inf values", flush=True)
                        
                        # Sample-wise analysis for first few samples
                        for i in range(min(3, kpts_pred_2d_cam.shape[0])):
                            sample_loss = torch.abs(kpts_pred_2d_cam[i] - keypoints_2d_gt_cam[i]).mean().item()
                            print(f"   Sample {i} (ID: {sample_ids[i] if sample_ids and i < len(sample_ids) else 'Unknown'}): loss={sample_loss:.2f}", flush=True)
                    
                    # Clip extreme loss values to prevent training instability
                    if self.loss_params.get('clip_2d_loss', True):
                        max_loss_value = self.loss_params.get('max_2d_loss_value', 70.0)
                        if loss_val_cam.item() > max_loss_value:
                            print(f"   🔧 Clipping loss from {loss_val_cam.item():.2f} to {max_loss_value}", flush=True)
                            loss_val_cam = torch.clamp(loss_val_cam, max=max_loss_value)
                    
                    total_2d_loss += loss_val_cam
                    n_cameras += 1
                    used_cameras.append(cam_name)

                if n_cameras > 0:
                    avg_2d_loss = total_2d_loss / n_cameras
                    total_loss.append(avg_2d_loss)
                    loss_dict[f"{loss_name}_2d"] = avg_2d_loss.detach().clone().cpu().item()

        if len(total_loss) == 0:
            # A fully masked 2D-only batch should no-op instead of crashing backward().
            return kpts_pred.sum() * 0.0, loss_dict

        return sum(total_loss), loss_dict

    @property
    def names(self):
        # Include both 3D and 2D loss names for CSV logging
        names = list(self.loss_fcns_3d.keys())
        names.extend([f"{name}_2d" for name in self.loss_fcns_2d.keys()])
        if self.loss_params.get("learned_visibility_enabled", False):
            names.append("VisibilityBCE")
        return names


def save_2d_reprojection_visualizations_old(
    epoch,
    batch_idx,
    volumes,
    kpts_pred,
    keypoints_2d_gt,
    cameras,
    params,
    checkpoint_dir,
    dataset=None,
    batch=None,
    max_cameras_to_plot=6,
):
    """
    Save coordinate comparison plots for GT vs predicted 2D keypoints.

    Creates scatter plots showing ground truth vs predicted 2D coordinates
    in their original image coordinate space. This helps debug:
    - Whether predictions are in the right scale/range
    - Whether there are systematic offsets
    - Whether the 2D reprojection loss is meaningful
    
    Note: For overlay visualization on actual images, you would need access
    to the DataGenerator to load the original cropped images that were used
    for training. This simplified version focuses on coordinate comparison.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"🔍 ENTERED VIS FUNCTION: epoch={epoch}, batch_idx={batch_idx}", flush=True)
    print(f"  kpts_pred: {kpts_pred.shape if kpts_pred is not None else None}", flush=True)
    print(f"  keypoints_2d_gt: {type(keypoints_2d_gt)}", flush=True)
    if isinstance(keypoints_2d_gt, dict):
        print(f"  keypoints_2d_gt keys: {list(keypoints_2d_gt.keys())}", flush=True)
    print(f"  cameras: {type(cameras)}", flush=True)
    if isinstance(cameras, dict):
        print(f"  cameras keys: {list(cameras.keys())}", flush=True)
    print(f"  dataset: {type(dataset)}", flush=True)
    print(f"  batch type: {type(batch)}", flush=True)

    # Guardrails - don't need volumes for coordinate comparison
    if kpts_pred is None or cameras is None:
        print(f"❌ EARLY EXIT: kpts_pred={kpts_pred is not None}, cameras={cameras is not None}", flush=True)
        return

    # Flatten possibly nested cameras dict
    flat_cameras = cameras
    try:
        if isinstance(cameras, dict) and len(cameras) > 0:
            sample_val = next(iter(cameras.values()))
            if isinstance(sample_val, dict) and ("K" not in sample_val):
                merged = {}
                for _exp_idx, cam_map in cameras.items():
                    if not isinstance(cam_map, dict):
                        continue
                    print(f"  Processing exp_idx: {_exp_idx}, cam_map keys: {list(cam_map.keys())}", flush=True)
                    for _cam_name, _cam_params in cam_map.items():
                        # Create prefixed camera name: exp_idx + "_" + cam_name
                        prefixed_name = f"{_exp_idx}_{_cam_name}"
                        merged[prefixed_name] = _cam_params
                        print(f"    Created camera mapping: {prefixed_name}", flush=True)
                flat_cameras = merged
                print(f"🔧 Flattened cameras: {list(flat_cameras.keys())}", flush=True)
    except Exception as e:
        print(f"⚠️ Camera flattening failed: {e}", flush=True)
        flat_cameras = cameras

    # Compute predicted 2D projections per camera using DANNCE's method
    kpts_pred_2d_dict = {}
    kpts_pred_np = kpts_pred.detach().cpu().numpy()  # [B, 3, n_keypoints]
    batch_size, _, n_keypoints = kpts_pred_np.shape
    
    try:
        for cam_name, cam_params in flat_cameras.items():
            # Extract camera parameters
            K = np.array(cam_params['K'])
            R = np.array(cam_params.get('R', cam_params.get('r')))
            t = np.array(cam_params['t'])
            if t.shape == (3,):
                t = t.reshape(1, 3)
            elif t.shape == (3, 1):
                t = t.T

            cam_projections = []
            for b in range(batch_size):
                # Reshape to (n_keypoints, 3) as expected by project_to_2d
                pts_3d = kpts_pred_np[b].T  # [n_keypoints, 3]
                
                # DEBUG: Show 3D input points for first batch
                if b == 0 and epoch <= 1:  # Only debug first batch and early epochs
                    print(f"    🔍 PROJECTION DEBUG - {cam_name}, batch {b}", flush=True)
                    print(f"      3D points (first 3): {pts_3d[:3]}", flush=True)
                    print(f"      Camera K shape: {K.shape}, R shape: {R.shape}, t shape: {t.shape}", flush=True)
                
                # Use DANNCE's project_to_2d function
                projpts = project_to_2d(pts_3d, K, R, t)[:, :2]  # [n_keypoints, 2]
                
                # DEBUG: Show projection results
                if b == 0 and epoch <= 1:
                    print(f"      Projected 2D (first 3): {projpts[:3]}", flush=True)
                
                # Apply lens distortion if available
                if 'RDistort' in cam_params and 'TDistort' in cam_params:
                    try:
                        projpts_distorted = distortPoints(
                            projpts,
                            K,
                            np.squeeze(cam_params['RDistort']),
                            np.squeeze(cam_params['TDistort']),
                        ).T  # Result is (2, n_keypoints), transpose to (n_keypoints, 2)
                        if b == 0 and epoch <= 1:
                            print(f"      Distorted 2D (first 3): {projpts_distorted[:3]}", flush=True)
                        projpts = projpts_distorted
                    except Exception as e:
                        if b == 0 and epoch <= 1:
                            print(f"      Distortion failed: {e}", flush=True)
                        pass  # Use undistorted points if distortion fails
                
                # Convert to (2, n_keypoints) format to match GT
                cam_projections.append(projpts.T)  # [2, n_keypoints]
            
            # Stack and convert to tensor: [B, 2, n_keypoints]
            kpts_pred_2d_dict[cam_name] = torch.from_numpy(np.stack(cam_projections)).to(kpts_pred.device)
            
    except Exception as e:
        kpts_pred_2d_dict = None

    # Prepare save directory
    vis_dir = os.path.join(checkpoint_dir or ".", "debug_vis", f"epoch{epoch}", f"batch{batch_idx}")
    os.makedirs(vis_dir, exist_ok=True)

    # Get batch info
    batch_size = kpts_pred.shape[0] if kpts_pred is not None else 1
    
    # Try to access actual images from dataset if available
    image_data = None
    if dataset is not None and hasattr(dataset, 'list_IDs'):
        try:
            # For NPY dataset, try to figure out current sample IDs
            print(f"  Dataset has list_IDs: {len(dataset.list_IDs)} samples", flush=True)
            print(f"  Dataset type: {type(dataset).__name__}", flush=True)
            
            # Check if dataset has image loading capabilities
            if hasattr(dataset, 'labels') and hasattr(dataset, 'load_frame'):
                print(f"  Dataset has image loading capabilities", flush=True)
                # This is a DataGenerator_3Dconv with video loading
                sample_id = dataset.list_IDs[batch_idx] if batch_idx < len(dataset.list_IDs) else dataset.list_IDs[0]
                print(f"  Trying to load images for sample: {sample_id}", flush=True)
                
                # Load images for each camera
                if "_" in sample_id:
                    experimentID = int(sample_id.split("_")[0])
                else:
                    experimentID = 0
                    
                image_data = {}
                try:
                    for cam_name in dataset.camnames[experimentID]:
                        frame_info = dataset.labels[sample_id]["frames"][cam_name]
                        print(f"    Loading {cam_name}: frame {frame_info}", flush=True)
                        
                        # Load the actual image using the dataset's loader
                        thisim = dataset.load_frame.load_vid_frame(
                            frame_info, cam_name, extension=dataset.extension,
                        )
                        
                        # Apply the same cropping that DANNCE uses
                        cropped_im = thisim[
                            dataset.crop_height[0] : dataset.crop_height[1],
                            dataset.crop_width[0] : dataset.crop_width[1],
                        ]
                        
                        # Store with prefixed name to match camera flattening
                        prefixed_name = f"{experimentID}_{cam_name}"
                        image_data[prefixed_name] = cropped_im
                        print(f"    Loaded {prefixed_name}: {cropped_im.shape}", flush=True)
                        
                except Exception as e:
                    print(f"  ❌ Error loading images: {e}", flush=True)
                    image_data = None
            else:
                print(f"  Dataset does not support image loading (NPY-based)", flush=True)
                
        except Exception as e:
            print(f"  ❌ Error accessing dataset: {e}", flush=True)

    # Define camera order to iterate
    cam_names_sorted = list(sorted(flat_cameras.keys())) if isinstance(flat_cameras, dict) else []
    print(f"📷 Available cameras: {cam_names_sorted}", flush=True)
    # Limit to available cams and to a reasonable number for plotting
    num_to_plot = min(len(cam_names_sorted), max_cameras_to_plot)

    # Create visualization for each camera
    for cam_idx in range(num_to_plot):
        if cam_idx >= len(cam_names_sorted):
            break
        cam_name = cam_names_sorted[cam_idx]
        
        # Check if we have actual image data for this camera
        if image_data is not None and cam_name in image_data:
            # Create image overlay visualization
            img = image_data[cam_name]
            print(f"  Using actual image for {cam_name}: {img.shape}", flush=True)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Normalize image for display
            if len(img.shape) == 3:
                # RGB image
                img_display = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img * 0.0
            else:
                # Grayscale - convert to RGB for consistent display
                img_norm = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img * 0.0
                img_display = np.stack([img_norm, img_norm, img_norm], axis=-1)
            
            ax.imshow(img_display, origin="upper")
            ax.set_title(f"2D Keypoint Overlay - {cam_name}")
            
            # Set coordinate system to match image pixels
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)  # Flip Y axis for image coordinates
            
        else:
            # Fallback to coordinate comparison plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"2D Coordinate Comparison - {cam_name}")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")

        print(f"\n🔍 COORD VIS - {cam_name}, epoch {epoch}", flush=True)
        print(f"  Looking for GT camera: {cam_name} in {list(keypoints_2d_gt.keys()) if isinstance(keypoints_2d_gt, dict) else 'No GT dict'}", flush=True)
        
        # Debug: Check if there's a close match
        if isinstance(keypoints_2d_gt, dict):
            gt_keys = list(keypoints_2d_gt.keys())
            print(f"  Debug: cam_name='{cam_name}', GT keys={gt_keys}", flush=True)
            # Try without the double prefix
            if cam_name.startswith('0_0_'):
                alt_name = cam_name.replace('0_0_', '0_', 1)
                print(f"  Trying alternative name: '{alt_name}'", flush=True)
                if alt_name in keypoints_2d_gt:
                    print(f"  ✅ Found GT data using alternative name: {alt_name}", flush=True)
                    cam_name = alt_name  # Use the correct name

        # Plot GT 2D keypoints (if available)
        if isinstance(keypoints_2d_gt, dict) and cam_name in keypoints_2d_gt:
            gt = keypoints_2d_gt[cam_name]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu()
            try:
                b = 0  # First batch
                gt_x_orig = gt[b, 0].numpy()
                gt_y_orig = gt[b, 1].numpy()
                
                if image_data is not None and cam_name in image_data:
                    # Apply DANNCE's coordinate transformation for image overlay
                    crop_width = params.get('crop_width', [0, 1280])
                    crop_height = params.get('crop_height', [0, 720])
                    downfac = params.get('downfac', 1) or 1
                    
                    # Apply the same transformation as DANNCE's debug_com function
                    gt_x = (gt_x_orig - crop_width[0]) / downfac
                    gt_y = (gt_y_orig - crop_height[0]) / downfac
                    print(f"  GT transformed: crop_width={crop_width}, crop_height={crop_height}, downfac={downfac}", flush=True)
                else:
                    # Use original coordinates for coordinate comparison
                    gt_x = gt_x_orig
                    gt_y = gt_y_orig
                
                valid_gt = ~(np.isnan(gt_x) | np.isnan(gt_y))
                
                ax.scatter(gt_x[valid_gt], gt_y[valid_gt], s=50, c="lime", marker="o", 
                          label=f"GT ({valid_gt.sum()}/{len(valid_gt)})", alpha=0.7)
                print(f"  GT coords range: x=[{gt_x[valid_gt].min():.1f}, {gt_x[valid_gt].max():.1f}], y=[{gt_y[valid_gt].min():.1f}, {gt_y[valid_gt].max():.1f}]", flush=True)
            except Exception as e:
                print(f"  GT error: {e}", flush=True)

        # Plot predicted 2D projections (if available)
        if kpts_pred_2d_dict is not None and isinstance(kpts_pred_2d_dict, dict) and cam_name in kpts_pred_2d_dict:
            pred = kpts_pred_2d_dict[cam_name]
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu()
            try:
                b = 0  # First batch
                pred_x_orig = pred[b, 0].numpy()
                pred_y_orig = pred[b, 1].numpy()
                
                if image_data is not None and cam_name in image_data:
                    # Apply DANNCE's coordinate transformation for image overlay
                    crop_width = params.get('crop_width', [0, 1280])
                    crop_height = params.get('crop_height', [0, 720])
                    downfac = params.get('downfac', 1) or 1
                    
                    # Apply the same transformation as DANNCE's debug_com function
                    pred_x = (pred_x_orig - crop_width[0]) / downfac
                    pred_y = (pred_y_orig - crop_height[0]) / downfac
                else:
                    # Use original coordinates for coordinate comparison
                    pred_x = pred_x_orig
                    pred_y = pred_y_orig
                
                valid_pred = ~(np.isnan(pred_x) | np.isnan(pred_y))
                
                ax.scatter(pred_x[valid_pred], pred_y[valid_pred], s=50, c="red", marker="x", 
                          label=f"Pred ({valid_pred.sum()}/{len(valid_pred)})", alpha=0.7)
                print(f"  Pred coords range: x=[{pred_x[valid_pred].min():.1f}, {pred_x[valid_pred].max():.1f}], y=[{pred_y[valid_pred].min():.1f}, {pred_y[valid_pred].max():.1f}]", flush=True)
            except Exception as e:
                print(f"  Pred error: {e}", flush=True)

        # Add legend and save
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(vis_dir, f"{cam_name}.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
        plt.close(fig)
        print(f"  Saved coordinate plot: {out_path}", flush=True)

    return



def save_2d_reprojection_visualizations(
    epoch,
    batch_idx,
    volumes,
    kpts_pred,
    keypoints_2d_gt,
    cameras,
    params,
    checkpoint_dir,
    dataset=None,
    batch=None,
    max_cameras_to_plot=6,
    sample_id=None, # Use single sample_id
):
    """
    Save visualizations with GT and predicted 2D keypoints overlaid on original video frames.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio
    import os
    from pathlib import Path
    from dannce.engine.utils.projection import project_to_2d, distortPoints

    # This function is now very verbose. Return early to disable.
    return

    print(f"🔍 ENTERED VIS FUNCTION: epoch={epoch}, batch_idx={batch_idx}", flush=True)
    print(f"  kpts_pred: {kpts_pred.shape if kpts_pred is not None else None}", flush=True)
    print(f"  keypoints_2d_gt: {type(keypoints_2d_gt)}", flush=True)
    if isinstance(keypoints_2d_gt, dict):
        print(f"  keypoints_2d_gt keys: {list(keypoints_2d_gt.keys())}", flush=True)
    print(f"  cameras: {type(cameras)}", flush=True)
    if isinstance(cameras, dict):
        print(f"  cameras keys: {list(cameras.keys())}", flush=True)
    print(f"  volumes: {volumes.shape if volumes is not None else None}", flush=True)

    # Guardrails
    if kpts_pred is None or cameras is None:
        print(f"❌ EARLY EXIT: kpts_pred={kpts_pred is not None}, cameras={cameras is not None}", flush=True)
        return

    # Flatten possibly nested cameras dict
    flat_cameras = cameras
    try:
        if isinstance(cameras, dict) and len(cameras) > 0:
            sample_val = next(iter(cameras.values()))
            if isinstance(sample_val, dict) and ("K" not in sample_val):
                merged = {}
                for _exp_idx, cam_map in cameras.items():
                    if not isinstance(cam_map, dict):
                        continue
                    for _cam_name, _cam_params in cam_map.items():
                        prefixed_name = f"{_exp_idx}_{_cam_name}"
                        merged[prefixed_name] = _cam_params
                flat_cameras = merged
    except Exception as e:
        print(f"⚠️ Camera flattening failed: {e}", flush=True)
        flat_cameras = cameras
    
    print(f"🔧 Flattened cameras: {list(flat_cameras.keys())}", flush=True)

    # Extract dimensions
    batch_size, n_dims, n_keypoints = kpts_pred.shape

    # Project 3D predictions to 2D
    kpts_pred_2d_dict = {}
    kpts_pred_np = kpts_pred.detach().cpu().numpy()  # [B, 3, n_keypoints]
    
    try:
        for cam_name, cam_params in flat_cameras.items():
            # Extract camera parameters
            K = np.array(cam_params['K'])
            R = np.array(cam_params.get('R', cam_params.get('r')))
            t = np.array(cam_params['t'])
            if t.shape == (3,):
                t = t.reshape(1, 3)
            elif t.shape == (3, 1):
                t = t.T

            cam_projections = []
            for b in range(batch_size):
                # Reshape to (n_keypoints, 3) as expected by project_to_2d
                pts_3d = kpts_pred_np[b].T  # [n_keypoints, 3]
                
                # Debug print for first batch
                if b == 0 and epoch <= 1:
                    print(f"    🔍 PROJECTION DEBUG - {cam_name}, batch {b}", flush=True)
                    print(f"      3D points (first 3): {pts_3d[:3]}", flush=True)
                    print(f"      Camera K shape: {K.shape}, R shape: {R.shape}, t shape: {t.shape}", flush=True)
                
                # Use DANNCE's project_to_2d function
                projpts = project_to_2d(pts_3d, K, R, t)[:, :2]  # [n_keypoints, 2]
                
                if b == 0 and epoch <= 1:
                    print(f"      Projected 2D (first 3): {projpts[:3]}", flush=True)
                
                # Apply lens distortion if available
                if 'RDistort' in cam_params and 'TDistort' in cam_params:
                    try:
                        projpts_distorted = distortPoints(
                            projpts,
                            K,
                            np.squeeze(cam_params['RDistort']),
                            np.squeeze(cam_params['TDistort']),
                        ).T  # Result is (2, n_keypoints), transpose to (n_keypoints, 2)
                        if b == 0 and epoch <= 1:
                            print(f"      Distorted 2D (first 3): {projpts_distorted[:3]}", flush=True)
                        projpts = projpts_distorted
                    except Exception as e:
                        if b == 0 and epoch <= 1:
                            print(f"      Distortion failed: {e}", flush=True)
                        pass  # Use undistorted points if distortion fails
                
                # Convert to (2, n_keypoints) format to match GT
                cam_projections.append(projpts.T)  # [2, n_keypoints]
            
            # Stack and convert to tensor: [B, 2, n_keypoints]
            kpts_pred_2d_dict[cam_name] = torch.from_numpy(np.stack(cam_projections)).to(kpts_pred.device)
            
    except Exception as e:
        print(f"❌ Projection error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        kpts_pred_2d_dict = None

    # Prepare save directory
    vis_dir = os.path.join(checkpoint_dir or ".", "debug_vis", f"epoch{epoch}", f"batch{batch_idx}")
    os.makedirs(vis_dir, exist_ok=True)

    # Get frame indices and video paths
    print(f"📷 Available cameras: {list(flat_cameras.keys())}", flush=True)
    
    # For each camera, create overlay visualization
    for cam_idx, (cam_name, cam_params) in enumerate(flat_cameras.items()):
        if cam_idx >= max_cameras_to_plot:
            break
            
        # Extract experiment index and actual camera name from flattened name
        # Handle various formats: "0_Camera1", "0_0_Camera1", "Camera1"
        parts = cam_name.split("_")
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            # Format: "0_0_Camera1" -> exp_idx=0, actual_cam_name="Camera1"
            exp_idx = int(parts[0])
            actual_cam_name = "_".join(parts[2:])
        elif len(parts) >= 2 and parts[0].isdigit():
            # Format: "0_Camera1" -> exp_idx=0, actual_cam_name="Camera1"
            exp_idx = int(parts[0])
            actual_cam_name = "_".join(parts[1:])
        else:
            # Format: "Camera1" -> exp_idx=0, actual_cam_name="Camera1"
            exp_idx = 0
            actual_cam_name = cam_name
            
        print(f"  🔍 Camera name parsing: '{cam_name}' -> exp_idx={exp_idx}, actual_cam_name='{actual_cam_name}'")
            
        print(f"\n🔍 DEBUG VIS - {cam_name}, batch {batch_idx}, epoch {epoch}", flush=True)
        
        try:
            # Load the original video frame
            frame = None
            
            print(f"  🔍 ENTRY DEBUG: dataset={dataset is not None}, has_list_IDs={hasattr(dataset, 'list_IDs') if dataset else False}", flush=True)
            
            # PROPER WORKFLOW: Follow dataset's video loading mechanism
            if dataset is not None and sample_id is not None:
                print(f"  🔍 ENTERING dataset handling block", flush=True)
                # sample_id is now passed directly
                print(f"  📋 Sample ID: {sample_id}", flush=True)
                
                # Debug dataset attributes
                try:
                    print(f"  🔍 Dataset type: {type(dataset).__name__}", flush=True)
                    print(f"  🔍 Has labels: {hasattr(dataset, 'labels')}", flush=True)
                    print(f"  🔍 Has labels_2d: {hasattr(dataset, 'labels_2d')}", flush=True)
                    print(f"  🔍 Has load_frame: {hasattr(dataset, 'load_frame')}", flush=True)
                    print(f"  🔍 Has vidreaders: {hasattr(dataset, 'vidreaders')}", flush=True)
                except Exception as debug_e:
                    print(f"  ❌ Debug error: {debug_e}", flush=True)
                
                # For NPY datasets, 2D data is in labels_2d, not labels
                frame_idx = None
                video_path = None
                
                print(f"  🔍 Checking labels_2d: hasattr={hasattr(dataset, 'labels_2d')}, not_none={getattr(dataset, 'labels_2d', None) is not None}", flush=True)
                
                if hasattr(dataset, 'labels_2d') and dataset.labels_2d is not None:
                    print(f"  📋 Using labels_2d for NPY dataset", flush=True)
                    if sample_id not in dataset.labels_2d:
                        print(f"  ❌ Sample {sample_id} not in dataset.labels_2d", flush=True)
                        print(f"  📋 Available sample IDs (first 5): {list(dataset.labels_2d.keys())[:5]}", flush=True)
                    else:
                        sample_2d_entry = dataset.labels_2d[sample_id]
                        print(f"  📋 Sample 2D entry keys: {list(sample_2d_entry.keys()) if isinstance(sample_2d_entry, dict) else 'Not a dict'}", flush=True)
                        
                        # Use the robust frame lookup function
                        frame_id = get_frame_id_from_sample(
                            {sample_id: sample_2d_entry}, 
                            sample_id, 
                            actual_cam_name, 
                            exp_idx
                        )
                        
                        # PROOF: Compare lookup result vs sample_id parsing
                        sample_id_parsed = int(sample_id.split("_")[1])
                        print(f"  🧪 FRAME INDEX VERIFICATION:", flush=True)
                        print(f"     Sample ID parsed: {sample_id_parsed}", flush=True)
                        print(f"     Frame lookup result: {frame_id}", flush=True)
                        print(f"     Using lookup: {'✅ YES' if frame_id is not None else '❌ NO (fallback)'}", flush=True)
                        if frame_id is not None and frame_id != sample_id_parsed:
                            print(f"     🎯 DIFFERENT VALUES! Proof we're using lookup, not sample_id parsing", flush=True)
                        elif frame_id is not None:
                            print(f"     ⚠️ Values match - can't visually distinguish lookup vs parsing", flush=True)
                        
                        if frame_id is None:
                            print(f"  ⚠️ Frame lookup failed, using fallback from sample_id", flush=True)
                            frame_id = sample_id_parsed
                            print(f"  ⚠️ Using fallback frame_id from sample_id: {frame_id}", flush=True)
                        
                        # Dump dataset-provided 2D coordinates for this camera/sample
                        try:
                            if "data" in sample_2d_entry:
                                cam_data_map = sample_2d_entry["data"]
                                # Try multiple key variants
                                ds_keys_to_try = [
                                    actual_cam_name,
                                    cam_name,
                                    f"{exp_idx}_{actual_cam_name}",
                                    f"0_{actual_cam_name}",
                                ]
                                ds_used_key = None
                                ds_arr = None
                                for ktry in ds_keys_to_try:
                                    if ktry in cam_data_map:
                                        ds_arr = cam_data_map[ktry]
                                        ds_used_key = ktry
                                        break
                                if ds_arr is not None:
                                    if torch.is_tensor(ds_arr):
                                        ds_arr = ds_arr.detach().cpu().numpy()
                                    ds_arr = np.array(ds_arr)
                                    print(f"  🧩 Dataset 2D using key '{ds_used_key}': shape={ds_arr.shape}", flush=True)
                                    print(f"  🧩 Dataset 2D coords (2 x n):\n{ds_arr}", flush=True)
                                    print(f"  🧩 Dataset 2D coords (n x 2):\n{ds_arr.T}", flush=True)
                                else:
                                    print(f"  ⚠️ Dataset 2D not found for keys {ds_keys_to_try}", flush=True)
                        except Exception as _ds_e:
                            print(f"  ⚠️ Dataset 2D dump failed: {_ds_e}", flush=True)

                        # Try to extract experiment info from sample_id to find video
                        if "_" in sample_id:
                            exp_id = int(sample_id.split("_")[0])
                            print(f"  📋 Parsed: exp_id={exp_id}, actual_frame_id={frame_id}", flush=True)
                            
                            # Try to use training parameters to find video paths
                            if 'exp' in params and len(params['exp']) > exp_id:
                                exp_config = params['exp'][exp_id]
                                print(f"  📋 Found experiment config: {list(exp_config.keys())}", flush=True)
                                
                                if 'viddir' in exp_config:
                                    viddir = exp_config['viddir']
                                    print(f"  📁 Video directory: {viddir}", flush=True)
                                    
                                    # Try to construct video path and load frame
                                    candidate_path = Path(viddir) / actual_cam_name / "0.mp4"
                                    if candidate_path.exists():
                                        print(f"  ✅ Found video: {candidate_path}", flush=True)
                                        video_path = candidate_path
                                        frame_idx = frame_id
                                    else:
                                        # Try alternative structures
                                        alt_paths = [
                                            Path(viddir) / actual_cam_name / f"{frame_id}.mp4",
                                            Path(viddir) / f"vid{exp_id+1}" / "videos" / actual_cam_name / "0.mp4",
                                        ]
                                        for alt_path in alt_paths:
                                            if alt_path.exists():
                                                print(f"  ✅ Found video: {alt_path}", flush=True)
                                                video_path = alt_path
                                                frame_idx = frame_id
                                                break
                                        else:
                                            print(f"  ❌ No video found at: {candidate_path}", flush=True)
                                            
                # Try to load frame if we found video path
                if frame_idx is not None and video_path is not None:
                    try:
                        print(f"  🎬 Loading frame {frame_idx} from {video_path}", flush=True)
                        print(f"  🎯 FINAL VERIFICATION: frame_idx={frame_idx} (this is what gets passed to video.get_frame())", flush=True)
                        from dannce.engine.data.video import MediaVideo
                        video = MediaVideo(filename=str(video_path), grayscale=False, bgr=True)
                        frame = video.get_frame(frame_idx, grayscale=False)
                        print(f"  ✅ Loaded frame via experiment config: shape={frame.shape}", flush=True)
                        print(f"  📊 Frame stats: dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}", flush=True)
                    except Exception as e:
                        print(f"  ❌ Failed to load frame from experiment config: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        frame = None
                                            
                elif hasattr(dataset, 'labels') and dataset.labels is not None:
                    # Original video dataset logic...
                    print(f"  📋 Using labels for video dataset", flush=True)
                    if sample_id and sample_id in dataset.labels:
                        # Re-implement frame loading logic here using the correct sample_id
                        if "frames" in dataset.labels[sample_id]:
                            frames_info = dataset.labels[sample_id]["frames"]
                            if actual_cam_name in frames_info:
                                frame_idx = frames_info[actual_cam_name]
                                # video_path needs to be determined from the dataset's vidreaders
                                if hasattr(dataset, 'vidreaders'):
                                    # This part is complex, requires looking up video file from frame_idx
                                    pass
                    # ... (original code for video datasets)
                else:
                    print(f"  ❌ No usable labels found", flush=True)
                    print(f"  ❌ dataset.labels_2d is None: {getattr(dataset, 'labels_2d', 'MISSING') is None}", flush=True)
                    print(f"  ❌ dataset.labels is None: {getattr(dataset, 'labels', 'MISSING') is None}", flush=True)
                
                # Try to use dataset's own video loading mechanism first
                if hasattr(dataset, 'load_frame') and frame_idx is not None:
                    try:
                        print(f"  🎬 Using dataset's video loader...", flush=True)
                        frame = dataset.load_frame.load_vid_frame(frame_idx, actual_cam_name)
                        print(f"  ✅ Loaded frame via dataset: shape={frame.shape}", flush=True)
                        print(f"  📊 Frame stats: dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}", flush=True)
                        
                        # Ensure proper format for matplotlib
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                            print(f"  🔧 Converted frame to uint8", flush=True)
                    except Exception as e:
                        print(f"  ❌ Dataset video loading failed: {e}", flush=True)
                        frame = None
                        
                # Fallback: try to access dataset's video readers
                elif hasattr(dataset, 'vidreaders') and frame_idx is not None:
                    try:
                        print(f"  🎬 Using dataset's vidreaders...", flush=True)
                        print(f"  📁 Available cameras in vidreaders: {list(dataset.vidreaders.keys())}", flush=True)
                        
                        # Look for camera in vidreaders
                        target_cam_key = None
                        for cam_key in dataset.vidreaders.keys():
                            if actual_cam_name in cam_key or cam_key.endswith(actual_cam_name):
                                target_cam_key = cam_key
                                break
                        
                        if target_cam_key:
                            print(f"  📹 Found camera key: {target_cam_key}", flush=True)
                            cam_videos = dataset.vidreaders[target_cam_key]
                            print(f"  📁 Available videos: {list(cam_videos.keys())[:3]}{'...' if len(cam_videos) > 3 else ''}", flush=True)
                            
                            # Use the chunks/frame logic from LoadVideoFrame
                            if hasattr(dataset, '_N_VIDEO_FRAMES') and target_cam_key in dataset._N_VIDEO_FRAMES:
                                chunks = dataset._N_VIDEO_FRAMES[target_cam_key]
                                cur_video_id = np.nonzero([c <= frame_idx for c in chunks])[0][-1]
                                cur_first_frame = chunks[cur_video_id]
                                fname = str(cur_first_frame) + ".mp4"  # Try mp4 first
                                frame_num = int(frame_idx - cur_first_frame)
                                
                                keyname = os.path.join(target_cam_key, fname)
                                if keyname in cam_videos:
                                    video_path = cam_videos[keyname]
                                    print(f"  📹 Loading from: {video_path}, frame {frame_num}", flush=True)
                                    
                                    # Load using MediaVideo
                                    from dannce.engine.data.video import MediaVideo
                                    video = MediaVideo(filename=video_path, grayscale=False, bgr=True)
                                    frame = video.get_frame(frame_num, grayscale=False)
                                    print(f"  ✅ Loaded frame via vidreaders: shape={frame.shape}", flush=True)
                    except Exception as e:
                        print(f"  ❌ Vidreaders loading failed: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        frame = None
            else:
                print(f"  ❌ Dataset is None or sample_id was not provided.", flush=True)
            
            print(f"  🔍 AFTER dataset logic: frame={frame is not None}", flush=True)
            
            # If we couldn't load a frame, create a blank one
            if frame is None:
                print(f"  ⚠️ Creating blank frame as fallback", flush=True)
                # Get image dimensions from params or use defaults
                width = params.get('raw_im_w', 1280)
                height = params.get('raw_im_h', 720)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                print(f"  Using blank frame: {width}x{height}")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Debug frame before display
            if frame is not None:
                print(f"  🖼️ About to display frame: shape={frame.shape}, dtype={frame.dtype}", flush=True)
                print(f"     Range: [{frame.min()}, {frame.max()}], mean={frame.mean():.1f}", flush=True)
                
                # Check if frame is actually all zeros despite having good stats
                if frame.mean() < 5:
                    print(f"  ⚠️ Frame appears black despite loading! Inspecting...", flush=True)
                    print(f"     Unique values: {np.unique(frame.ravel())[:10]}", flush=True)
                    print(f"     Non-zero pixels: {np.count_nonzero(frame)}/{frame.size}", flush=True)
                
                # Save a test frame to verify the frame data is correct
                test_frame_path = os.path.join(vis_dir, f"test_frame_{cam_name}.png")
                try:
                    # Save using matplotlib directly
                    plt.imsave(test_frame_path, frame)
                    print(f"  💾 Saved test frame to: {test_frame_path}", flush=True)
                except Exception as e:
                    print(f"  ❌ Failed to save test frame: {e}", flush=True)
                
                # Additional check: verify frame right before imshow
                print(f"  🔍 RIGHT BEFORE imshow: mean={frame.mean():.1f}, shape={frame.shape}", flush=True)
            
            # Display the frame
            if frame is not None:
                try:
                    ax.imshow(frame)
                    print(f"  🖼️ Frame displayed with ax.imshow()", flush=True)
                    
                    # Check if matplotlib is somehow corrupting the display
                    print(f"  🔍 After imshow: ax limits = x{ax.get_xlim()}, y{ax.get_ylim()}", flush=True)
                except Exception as imshow_e:
                    print(f"  ❌ ax.imshow() failed: {imshow_e}", flush=True)
            else:
                print(f"  ⚠️ Frame is None, creating blank background", flush=True)
            
            # Plot GT keypoints if available - use robust lookup
            gt_2d, gt_key_used = get_gt_camera_key(keypoints_2d_gt, actual_cam_name, exp_idx)
            
            if gt_2d is not None:
                print(f"  📍 Found GT data using key: '{gt_key_used}'", flush=True)
                if isinstance(gt_2d, torch.Tensor):
                    gt_2d = gt_2d.detach().cpu().numpy()
                
                # Take first item in batch
                if len(gt_2d.shape) == 3:
                    gt_2d = gt_2d[0]  # Shape: [2, n_keypoints]

                # Dump full GT coords for inspection
                try:
                    print(f"  🧩 GT 2D [{gt_key_used}] shape: {gt_2d.shape}", flush=True)
                    print(f"  🧩 GT 2D coords (2 x n):\n{gt_2d}", flush=True)
                    print(f"  🧩 GT 2D coords (n x 2):\n{gt_2d.T}", flush=True)
                except Exception as _gt_dump_e:
                    print(f"  ⚠️ GT 2D dump failed: {_gt_dump_e}", flush=True)

                # Compare GT dict to dataset 2D if available
                try:
                    if dataset is not None and hasattr(dataset, 'labels_2d') and dataset.labels_2d is not None and sample_id is not None:
                        if sample_id in dataset.labels_2d and "data" in dataset.labels_2d[sample_id]:
                            ds_map = dataset.labels_2d[sample_id]["data"]
                            ds_arr = None
                            for ktry in [actual_cam_name, cam_name, f"{exp_idx}_{actual_cam_name}", f"0_{actual_cam_name}"]:
                                if ktry in ds_map:
                                    ds_arr = ds_map[ktry]
                                    break
                            if ds_arr is not None:
                                ds_arr = np.array(ds_arr)
                                if ds_arr.shape == gt_2d.shape:
                                    diff = np.nanmax(np.abs(ds_arr - gt_2d))
                                    print(f"  🔎 GT vs dataset 2D: max|diff|={diff:.6f}", flush=True)
                                else:
                                    print(f"  🔎 Shape mismatch (GT {gt_2d.shape} vs dataset {ds_arr.shape}), skipping diff.", flush=True)
                except Exception as _cmp_e:
                    print(f"  ⚠️ GT vs dataset compare failed: {_cmp_e}", flush=True)
                
                # Plot valid GT points
                valid_mask = ~np.isnan(gt_2d[0]) & ~np.isnan(gt_2d[1])
                if valid_mask.sum() > 0:
                    gt_x = gt_2d[0, valid_mask]
                    gt_y = gt_2d[1, valid_mask]
                    ax.scatter(gt_x, gt_y, c='red', s=100, marker='o', alpha=0.8, 
                              edgecolors='white', linewidth=2, label='GT 2D')
                    print(f"  📍 GT points: {valid_mask.sum()}/{len(valid_mask)}, "
                          f"x range: [{gt_x.min():.1f}, {gt_x.max():.1f}], "
                          f"y range: [{gt_y.min():.1f}, {gt_y.max():.1f}]")
                else:
                    print(f"  ⚠️ GT data found but all points are NaN")
            else:
                # GT lookup debug is handled by get_gt_camera_key function
                pass
            
            # Plot projected predictions if available
            if kpts_pred_2d_dict and cam_name in kpts_pred_2d_dict:
                pred_2d = kpts_pred_2d_dict[cam_name]
                if isinstance(pred_2d, torch.Tensor):
                    pred_2d = pred_2d.detach().cpu().numpy()
                
                # Take first item in batch
                if len(pred_2d.shape) == 3:
                    pred_2d = pred_2d[0]  # Shape: [2, n_keypoints]
                
                # Plot all predicted points (they should all be valid)
                pred_x = pred_2d[0]
                pred_y = pred_2d[1]
                ax.scatter(pred_x, pred_y, c='blue', s=100, marker='x', alpha=0.8,
                          linewidth=3, label='Predicted 2D')
                print(f"  Pred points: {len(pred_x)}, "
                      f"x range: [{pred_x.min():.1f}, {pred_x.max():.1f}], "
                      f"y range: [{pred_y.min():.1f}, {pred_y.max():.1f}]")
                
                # Draw lines connecting GT and predictions if both exist
                if gt_2d is not None:  # Use the GT data we found earlier
                    valid_mask = ~np.isnan(gt_2d[0]) & ~np.isnan(gt_2d[1])
                    for i in np.where(valid_mask)[0]:
                        ax.plot([gt_2d[0, i], pred_2d[0, i]], 
                               [gt_2d[1, i], pred_2d[1, i]], 
                               'gray', alpha=0.3, linewidth=1)
            
            # Set title and labels
            ax.set_title(f'{cam_name} - Epoch {epoch}, Batch {batch_idx}', fontsize=14)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.legend(loc='upper right')
            
            # Set axis limits to frame size
            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)  # Invert y-axis for image coordinates
            
            # Save figure
            out_path = os.path.join(vis_dir, f"{cam_name}.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Visualization saved to: {out_path}")
            
        except Exception as e:
            print(f"❌ Failed to create visualization for {cam_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"✅ VISUALIZATION COMPLETED for batch {batch_idx}")
    return


class MetricHelper:
    def __init__(self, params):
        self.metric_names = params["metric"]
        self._get_metrics()

    def _get_metrics(self):
        self.metrics = {}
        for met in self.metric_names:
            self.metrics[met] = getattr(custom_metrics, met)

    def evaluate(self, kpts_gt, kpts_pred):
        # perform NaN masking ONCE before metric computation
        metric_dict = {}
        if len(self.metric_names) == 0:
            return metric_dict

        kpts_pred, kpts_gt = self.mask_nan(kpts_pred, kpts_gt)
        for met in self.metric_names:
            metric_dict[met] = self.metrics[met](kpts_pred, kpts_gt)

        return metric_dict

    @property
    def names(self):
        return self.metric_names

    @classmethod
    def mask_nan(self, pred, gt):
        """
        pred, gt: [bs, 3, n_joints]
        """
        pred = np.transpose(pred.copy(), (1, 0, 2))
        gt = np.transpose(gt.copy(), (1, 0, 2))  # [3, bs, n_joints]
        pred = np.reshape(pred, (pred.shape[0], -1))
        gt = np.reshape(gt, (gt.shape[0], -1))  # [3, bs*n_joints]

        gi = np.where(~np.isnan(np.sum(gt, axis=0)))[0]  # [bs*n_joints]

        pred = pred[:, gi]
        gt = gt[:, gi]  # [3, bs*n_joints]

        return pred, gt
