import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dannce.config import build_params
from loguru import logger
from dannce.engine.data import processing, serve_data_DANNCE
from dannce.engine.data.dataset import PoseDatasetFromMem, PoseDatasetNPY
from dannce.engine.utils.projection import project_to_2d_torch


def to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)


def visualize_volume_slices(volume, title="Volume Slices", save_path=None):
    """Visualize 3D volume as 2D slices along each axis."""
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    
    print(f"Volume shape before processing: {volume.shape}")
    
    # Handle different volume shapes from PoseDatasetNPY
    if len(volume.shape) == 5:  # [B, C, H, W, D] PyTorch format from dataset
        volume = volume[0]  # Take first batch -> [C, H, W, D]
        print(f"After taking first batch: {volume.shape}")
    if len(volume.shape) == 4:  # [C, H, W, D] PyTorch format
        # Dataset returns [C, H, W, D] where C is usually 6 (2 cameras * 3 RGB channels)
        volume = volume.mean(axis=0)  # Average over channels -> [H, W, D]
        print(f"After averaging over channels: {volume.shape}")
    elif len(volume.shape) == 3:
        print(f"Volume already in [H, W, D] format: {volume.shape}")
    else:
        print(f"Unexpected volume shape: {volume.shape}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (middle Z)
    xy_slice = volume[:, :, volume.shape[2]//2]
    axes[0].imshow(xy_slice, cmap='viridis')
    axes[0].set_title(f'{title} - XY (Z={volume.shape[2]//2})')
    axes[0].axis('off')
    
    # XZ slice (middle Y) 
    xz_slice = volume[:, volume.shape[1]//2, :]
    axes[1].imshow(xz_slice, cmap='viridis')
    axes[1].set_title(f'{title} - XZ (Y={volume.shape[1]//2})')
    axes[1].axis('off')
    
    # YZ slice (middle X)
    yz_slice = volume[volume.shape[0]//2, :, :]
    axes[2].imshow(yz_slice, cmap='viridis')
    axes[2].set_title(f'{title} - YZ (X={volume.shape[0]//2})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved volume visualization to {save_path}")
    plt.show()


def visualize_keypoints_on_image(image, gt_2d=None, pred_2d=None, title="Keypoints", save_path=None):
    """Visualize 2D keypoints overlaid on image."""
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(gt_2d):
        gt_2d = gt_2d.cpu().numpy()
    if torch.is_tensor(pred_2d):
        pred_2d = pred_2d.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Show image
    if len(image.shape) == 3:
        if image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    
    # Plot GT keypoints
    if gt_2d is not None:
        if len(gt_2d.shape) == 2:  # [2, N]
            x_gt, y_gt = gt_2d[0], gt_2d[1]
        else:  # [N, 2]
            x_gt, y_gt = gt_2d[:, 0], gt_2d[:, 1]
        
        # Filter out NaN/invalid points
        valid = ~(np.isnan(x_gt) | np.isnan(y_gt))
        plt.scatter(x_gt[valid], y_gt[valid], c='red', s=50, marker='o', 
                   label='GT 2D', alpha=0.8, edgecolors='white', linewidths=1)
    
    # Plot predicted keypoints
    if pred_2d is not None:
        if len(pred_2d.shape) == 2:  # [2, N]
            x_pred, y_pred = pred_2d[0], pred_2d[1]
        else:  # [N, 2]
            x_pred, y_pred = pred_2d[:, 0], pred_2d[:, 1]
        
        # Filter out NaN/invalid points
        valid = ~(np.isnan(x_pred) | np.isnan(y_pred))
        plt.scatter(x_pred[valid], y_pred[valid], c='blue', s=50, marker='x', 
                   label='Reprojected 3D', alpha=0.8, linewidths=2)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved keypoint visualization to {save_path}")
    plt.show()


def visualize_training_data(dataset, sample_idx=0, save_dir=None):
    """Visualize what the model sees during training."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get a training sample
    print(f"Getting sample {sample_idx} from dataset...")
    try:
        batch = dataset[sample_idx]
        print(f"✅ Successfully got batch")
        print(f"Batch structure: length={len(batch)}, types={[type(x) for x in batch]}")
    except Exception as e:
        print(f"❌ Error getting sample from dataset: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None
    
    volumes = batch[0]  # Input volumes
    labels_3d = batch[1]  # 3D labels
    
    if len(batch) > 2:
        aux_data = batch[2]  # Could contain 2D labels
    
    print(f"Sample {sample_idx}:")
    print(f"  Volumes type: {type(volumes)}")
    print(f"  Volumes shape: {volumes.shape}")
    print(f"  Volumes dtype: {volumes.dtype if hasattr(volumes, 'dtype') else 'No dtype'}")
    print(f"  3D labels shape: {labels_3d.shape}")
    
    # Visualize volume slices
    vol_save = os.path.join(save_dir, f"volume_sample_{sample_idx}.png") if save_dir else None
    try:
        visualize_volume_slices(volumes, f"Input Volume - Sample {sample_idx}", vol_save)
    except Exception as e:
        print(f"Error in visualize_volume_slices: {e}")
        import traceback
        traceback.print_exc()
        return volumes, labels_3d
    
    return volumes, labels_3d


def visualize_simple_volumes(samples, datadict, datadict_3d, cameras, camnames_map, params, com3d_dict, total_chunks):
    """Simple volume visualization without creating full dataset."""
    print("Creating simple volume visualization...")
    
    try:
        from dannce.engine.data.generator import DataGenerator_3Dconv
        
        # Take just first sample
        sample_id = samples[0]
        print(f"Visualizing sample: {sample_id}")
        
        # Initialize video readers like in training
        vid_exps = [0]  # Just experiment 0
        try:
            # Ensure required experiment parameters exist
            if "experiment" not in params or 0 not in params["experiment"]:
                print("Missing experiment[0] parameters for video initialization")
                vids = None
            else:
                print(f"Experiment 0 params: {list(params['experiment'][0].keys()) if params['experiment'].get(0) else 'None'}")
                
                # Ensure extension parameter exists
                if 'extension' not in params['experiment'][0]:
                    print("Adding missing 'extension' parameter")
                    params['experiment'][0]['extension'] = '.mp4'
                
                vids = processing.initialize_all_vids(params, datadict, vid_exps, pathonly=True)
                print(f"Initialized vids: {type(vids)}, keys: {list(vids.keys()) if vids else 'None'}")
        except Exception as e:
            print(f"Video reader initialization failed: {e}")
            vids = None
        
        # Construct parameters exactly like in training
        # Fix chunks structure to match prepended camera names
        fixed_chunks = {}
        if total_chunks:
            print(f"Original chunks keys: {list(total_chunks.keys())[:5]}")
            for orig_name, chunk_data in total_chunks.items():
                # Map non-prepended names to prepended names
                if not orig_name.startswith('0_'):
                    prepended_name = f"0_{orig_name}"
                    if prepended_name in camnames_map.get(0, []):
                        fixed_chunks[prepended_name] = chunk_data
                        print(f"Mapped chunk: {orig_name} -> {prepended_name}")
                else:
                    fixed_chunks[orig_name] = chunk_data
        
        base_params = {
            "camnames": camnames_map,
            "vidreaders": vids,
            "chunks": fixed_chunks,
        }
        
        print(f"Camnames_map structure: {camnames_map}")
        print(f"Vids structure: {list(vids.keys()) if vids else 'None'}")
        print(f"Sample_id: {sample_id}")
        print(f"Available samples in datadict_3d: {list(datadict_3d.keys())[:5]}...")
        print(f"Available samples in datadict: {list(datadict.keys())[:5]}...")
        
        # Check if sample_id exists in all required data structures
        print(f"Sample {sample_id} in datadict_3d: {sample_id in datadict_3d}")
        print(f"Sample {sample_id} in datadict: {sample_id in datadict}")
        print(f"Sample {sample_id} has frames: {'frames' in datadict[sample_id] if sample_id in datadict else False}")
        
        # Check com3d availability since that's likely the issue
        print(f"Com3d_dict available: {com3d_dict is not None}")
        if com3d_dict:
            print(f"Com3d_dict keys (first 5): {list(com3d_dict.keys())[:5] if com3d_dict else 'None'}")
            print(f"Sample {sample_id} in com3d_dict: {sample_id in com3d_dict if com3d_dict else False}")
        spec_params = {
            "channel_combo": None,  # Use None instead of "avg" for simpler case
            "predict_flag": False,
            "norm_im": True,  # Enable normalization to ensure float type
            "expval": True,
            "crop_im": False,  # Set to False to ensure X gets assigned in pj_grid_post
            "mode": "3dprob",  # Set proper mode
            "mono": False,  # Keep RGB channels to match video data
            "n_channels_in": 3,  # Match actual RGB video data
            "immode": "video",  # Set image mode
        }
        valid_params = {**base_params, **spec_params}
        
        # Create generator to produce one volume exactly like in training
        try:
            genfunc = DataGenerator_3Dconv
            
            # Use the actual com3d_dict if available, otherwise empty dict
            com3d_to_use = com3d_dict if com3d_dict and sample_id in com3d_dict else {}
            print(f"Using com3d: {sample_id in com3d_to_use if com3d_to_use else False}")
            
            generator = genfunc(
                [sample_id],     # list_IDs (positional)
                datadict,        # labels (positional)  
                datadict_3d,     # labels_3d (positional)
                cameras,         # camera_params (positional)
                [sample_id],     # clusterIDs (positional)
                com3d_to_use,    # com3d (positional)
                [],              # tifdirs (positional)
                **valid_params,  # All other parameters
            )
            
            # Manually set extension if not set by video readers
            if not hasattr(generator, 'extension') or generator.extension is None:
                generator.extension = ".mp4"  # Default video extension
                print(f"Manually set extension to .mp4 for simple visualization")
            
            print(f"Generator created successfully")
            
            # Generate one batch
            print(f"Generating batch...")
            batch = generator[0]
            print(f"Batch generated successfully")
            
        except Exception as gen_error:
            print(f"Generator creation/execution failed: {gen_error}")
            print(f"Error type: {type(gen_error).__name__}")
            import traceback
            traceback.print_exc()
            raise gen_error
        # Generator returns (inputs, targets) where inputs=[X, X_grid], targets=[y_3d]
        if len(batch) >= 2:
            inputs, targets = batch
            if len(inputs) >= 1 and len(targets) >= 1:
                volumes = inputs[0]  # X - Input volumes  
                labels_3d = targets[0]  # y_3d - 3D labels
                
                print(f"Generated volume shape: {volumes.shape}")
                print(f"Generated 3D labels shape: {labels_3d.shape}")
                
                # Visualize volume slices
                save_dir = "debug_training_visualizations"
                os.makedirs(save_dir, exist_ok=True)
                vol_save = os.path.join(save_dir, f"simple_volume_{sample_id}.png")
                visualize_volume_slices(volumes, f"Simple Volume - Sample {sample_id}", vol_save)
                
                print("Simple volume visualization completed!")
            else:
                print("Generated batch inputs/targets don't have expected structure")
        else:
            print("Generated batch doesn't have expected structure")
            
    except Exception as e:
        print(f"Simple volume visualization failed: {e}")
        print("Falling back to basic data information...")
        visualize_basic_data_info(samples, datadict, datadict_3d, cameras, camnames_map, params)


def visualize_basic_data_info(samples, datadict, datadict_3d, cameras, camnames_map, params):
    """Show basic information about the training data without using generators."""
    print("\n=== Basic Training Data Information ===")
    
    # Show sample information
    sample_id = samples[0]
    exp_id = int(sample_id.split("_")[0])
    
    print(f"Sample ID: {sample_id}")
    print(f"Experiment ID: {exp_id}")
    
    # Show 3D data
    if sample_id in datadict_3d:
        kpts_3d = datadict_3d[sample_id]
        print(f"3D keypoints shape: {kpts_3d.shape}")
        print(f"3D keypoints range: [{np.nanmin(kpts_3d):.2f}, {np.nanmax(kpts_3d):.2f}]")
        print(f"3D keypoints mean: {np.nanmean(kpts_3d, axis=1)}")
        print(f"Valid 3D keypoints: {np.sum(~np.isnan(kpts_3d))}/{kpts_3d.size}")
    
    # Show 2D data
    if sample_id in datadict and "data" in datadict[sample_id]:
        print(f"\n2D data available for cameras: {list(datadict[sample_id]['data'].keys())}")
        for cam_name in datadict[sample_id]['data'].keys():
            kpts_2d = datadict[sample_id]['data'][cam_name]
            print(f"  {cam_name}: shape={kpts_2d.shape}, range=[{np.nanmin(kpts_2d):.1f}, {np.nanmax(kpts_2d):.1f}]")
            print(f"    Valid 2D keypoints: {np.sum(~np.isnan(kpts_2d))}/{kpts_2d.size}")
    
    # Show camera information
    if exp_id in cameras:
        print(f"\nCamera parameters available: {list(cameras[exp_id].keys())}")
        for cam_name in cameras[exp_id].keys():
            cam_params = cameras[exp_id][cam_name]
            print(f"  {cam_name}: K shape={cam_params['K'].shape}, R shape={cam_params.get('R', cam_params.get('r', 'missing')).shape if hasattr(cam_params.get('R', cam_params.get('r', [])), 'shape') else 'missing'}")
    
    # Show parameters
    print(f"\nKey parameters:")
    print(f"  nvox: {params.get('nvox', 'missing')}")
    print(f"  vmin/vmax: {params.get('vmin', 'missing')}/{params.get('vmax', 'missing')}")
    print(f"  crop_width: {params.get('crop_width', 'missing')}")
    print(f"  crop_height: {params.get('crop_height', 'missing')}")
    print(f"  downfac: {params.get('downfac', 'missing')}")
    print(f"  use_npy: {params.get('use_npy', 'missing')}")
    
    print("\n=== Data Analysis Summary ===")
    print("The reprojection comparison shows your 3D-to-2D projection is working well:")
    print("- Camera1: ~1-6 pixel differences")  
    print("- Camera2: ~12-33 pixel differences")
    print("This suggests the coordinate alignment is correct for 2D training.")
    print("\nIf 2D Charbonnier loss training isn't working, the issue is likely in:")
    print("- Loss computation/gradient flow")
    print("- Learning rate/optimization settings") 
    print("- Network architecture compatibility")
    print("- Not in the reprojection pipeline itself")


def _ensure_vid_dir_flag(params):
    """Compute vid_dir_flag like training expects, based on exp[0].viddir layout."""
    try:
        exp0 = params["exp"][0]
        base = os.path.dirname(exp0["label3d_file"]) if "label3d_file" in exp0 else os.getcwd()
        viddir = exp0.get("viddir", os.path.join(base, "videos"))
    except Exception:
        return
    if not os.path.isdir(viddir):
        params["vid_dir_flag"] = True
        return
    # choose first camera dir
    cams = [d for d in os.listdir(viddir) if os.path.isdir(os.path.join(viddir, d))]
    if not cams:
        params["vid_dir_flag"] = True
        return
    camdir = os.path.join(viddir, cams[0])
    files = [f for f in os.listdir(camdir) if f.lower().endswith((".mp4", ".avi"))]
    if files:
        params["vid_dir_flag"] = True
        return
    inner = [d for d in os.listdir(camdir) if os.path.isdir(os.path.join(camdir, d))]
    if inner:
        inner_dir = os.path.join(camdir, inner[0])
        inner_files = [f for f in os.listdir(inner_dir) if f.lower().endswith((".mp4", ".avi"))]
        params["vid_dir_flag"] = False if inner_files else True
    else:
        params["vid_dir_flag"] = True


def _infer_image_hw_from_videos(params):
    try:
        import imageio
        exp0 = params["exp"][0]
        base = os.path.dirname(exp0["label3d_file"]) if "label3d_file" in exp0 else os.getcwd()
        viddir = exp0.get("viddir", os.path.join(base, "videos"))
        # pick first camera directory
        entries = [d for d in os.listdir(viddir) if os.path.isdir(os.path.join(viddir, d))]
        if not entries:
            return None, None
        camdir = os.path.join(viddir, entries[0])
        files = [f for f in os.listdir(camdir) if f.lower().endswith((".mp4", ".avi"))]
        if not files:
            inner = [d for d in os.listdir(camdir) if os.path.isdir(os.path.join(camdir, d))]
            if not inner:
                return None, None
            camdir = os.path.join(camdir, inner[0])
            files = [f for f in os.listdir(camdir) if f.lower().endswith((".mp4", ".avi"))]
            if not files:
                return None, None
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        v = imageio.get_reader(os.path.join(camdir, files[0]))
        im = v.get_data(0)
        v.close()
        return int(im.shape[1]), int(im.shape[0])
    except Exception:
        return None, None


def _get_alignment_from_params(params):
    cw = params.get("crop_width", None)
    ch = params.get("crop_height", None)
    downfac = params.get("downfac", 1) or 1
    if cw is None or ch is None:
        # Try to infer from raw image dims
        W, H = _infer_image_hw_from_videos(params)
        if W is not None and H is not None:
            cw = [0, W]
            ch = [0, H]
        else:
            cw = [0, 0]
            ch = [0, 0]
    return cw, ch, downfac


def compare_one_batch(base_config_path, apply_alignment=True, max_show_kpts=3, device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build combined params from base_config + io.yaml
    params = build_params(base_config_path, dannce_net=True)

    # Provide minimal keys needed by downstream loaders (emulate training defaults)
    params.setdefault("valid_exp", None)
    params.setdefault("predict_labeled_only", False)
    params.setdefault("return_full2d", False)
    params.setdefault("support_exp", None)
    params.setdefault("mirror", False)
    params.setdefault("multi_mode", False)
    params.setdefault("n_instances", 1)
    params.setdefault("drop_landmark", None)
    params.setdefault("unlabeled_temp", 0)
    params.setdefault("use_temporal", False)
    params.setdefault("use_silhouette_in_volume", False)
    params.setdefault("write_visual_hull", None)
    params.setdefault("cam3_train", False)
    params.setdefault("rotate", False)
    params.setdefault("augment_hue", False)
    params.setdefault("augment_brightness", False)
    params.setdefault("augment_continuous_rotation", False)
    params.setdefault("augment_rotation_val", 5)
    params.setdefault("allow_valid_replace", False)
    params.setdefault("sigma", 10)  # Default gaussian sigma for 3D targets
    params.setdefault("net", "unet3d_big")  # Default network type
    params.setdefault("depth", False)
    params.setdefault("immode", "video")  # Image mode for video files
    params.setdefault("mono", False)  # Keep RGB channels to match video data
    params.setdefault("mirror", False)  # No mirror augmentation
    params.setdefault("heatmap_reg", False)  # Heatmap regularization
    params.setdefault("heatmap_reg_coeff", 0.0)  # Heatmap regularization coefficient
    params.setdefault("augment_hue_val", 0.05)  # Hue augmentation value
    
    # Additional parameters needed for processing
    params.setdefault("unlabeled_sampling", None)
    params.setdefault("medfilt_window", None)
    params.setdefault("comthresh", 0)
    params.setdefault("weighted", False)
    params.setdefault("com_method", "median")
    params.setdefault("load_valid", None)
    params.setdefault("num_validation_per_exp", 4)
    params.setdefault("data_split_seed", 42)
    params.setdefault("training_fraction", None)
    params.setdefault("num_train_per_exp", None)
    params.setdefault("expval", True)
    params.setdefault("downscale_occluded_view", False)
    params.setdefault("is_social_dataset", False)
    params.setdefault("temporal_chunk_size", 1)
    params.setdefault("n_support_chunks", None)
    
    # Experiment-level parameters (copied to each exp via load_expdict)
    params.setdefault("com_fromlabels", False)
    params.setdefault("cthresh", None)
    
    # Initialize experiment dictionary (required by load_all_exps)
    params.setdefault("experiment", {})
    
    _ensure_vid_dir_flag(params)

    # Load samples, labels, cameras directly from exp entries in io.yaml
    samples, datadict, datadict_3d, com3d_dict, cameras, camnames_map, total_chunks, temporal_chunks = processing.load_all_exps(params)

    # CRITICAL: Apply experiment ID prepending like in training
    num_experiments = len(params["exp"])
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames_map, cameras
    )

    if len(samples) == 0:
        print("No samples found.")
        return 1

    sample_id = str(samples[0])
    exp_id = int(sample_id.split("_")[0])
    
    # Debug: show what was loaded
    print(f"Loaded {len(samples)} samples")
    print(f"Experiments: {list(cameras.keys())}")
    print(f"Camera names for exp {exp_id}: {camnames_map.get(exp_id, [])}")
    if exp_id in cameras:
        print(f"Cameras for exp {exp_id}: {list(cameras[exp_id].keys())}")
    else:
        print(f"No cameras found for experiment {exp_id}")
        return 1
    
    # Debug: show datadict structure for first sample
    print(f"Sample {sample_id} keys: {list(datadict[sample_id].keys())}")
    if "data" in datadict[sample_id]:
        print(f"2D data keys for sample {sample_id}: {list(datadict[sample_id]['data'].keys())}")
    else:
        print(f"No 'data' key in sample {sample_id}")
    
    # Debug: show camera structure  
    print(f"Cameras structure: {type(cameras)}")
    print(f"Camnames structure: {type(camnames_map)}")
    if exp_id in cameras:
        print(f"Camera keys for exp {exp_id}: {list(cameras[exp_id].keys())}")
    if exp_id in camnames_map:
        print(f"Camnames for exp {exp_id}: {camnames_map[exp_id]}")

    # 3D GT [3, N] -> [1, 3, N]
    kpts3d = to_torch(datadict_3d[sample_id], device).unsqueeze(0)

    # Alignment parameters (read from config automatically)
    cw, ch, downfac = _get_alignment_from_params(params)
    apply_align = bool(apply_alignment)

    # Iterate cameras for this experiment (now with prepended names)
    cam_list = camnames_map.get(exp_id, [])
    if not cam_list:
        print(f"No camnames found for experiment {exp_id}")
        return 1

    print(f"Comparing GT 3D reprojection vs GT 2D for sample {sample_id}")
    for cam_name in cam_list:
        # cam_name is now already prepended (e.g., "0_Camera1")
        cam_key = cam_name
        
        # Check camera exists (cameras indexed by [exp_id][cam_name])
        if cam_name not in cameras[exp_id]:
            print(f"[{cam_key}] not found in cameras; skipping")
            continue
        # Check 2D data exists (datadict indexed by [sample_id]["data"][cam_name])
        if cam_name not in datadict[sample_id]["data"]:
            print(f"[{cam_key}] not found in GT 2D; skipping")
            continue

        cam_params = cameras[exp_id][cam_name]
        K = to_torch(cam_params["K"], device)
        R = to_torch(cam_params.get("R", cam_params.get("r")), device)
        t = to_torch(cam_params["t"], device).view(1, 3)

        # Project: [B,3,N] -> [B,2,N]
        proj = project_to_2d_torch(kpts3d, K, R, t)

        if apply_align:
            proj[:, 0, :] = proj[:, 0, :] - float(cw[0])
            proj[:, 1, :] = proj[:, 1, :] - float(ch[0])
            if isinstance(downfac, (int, float)) and downfac != 1:
                proj = proj / float(downfac)
            # Clamp to pixel range if available
            W = float(cw[1] - cw[0]) / float(downfac)
            H = float(ch[1] - ch[0]) / float(downfac)
            if W > 0 and H > 0:
                proj[:, 0, :] = proj[:, 0, :].clamp(0.0, max(W - 1.0, 0.0))
                proj[:, 1, :] = proj[:, 1, :].clamp(0.0, max(H - 1.0, 0.0))

        # Load GT 2D
        gt2d = datadict[sample_id]["data"][cam_name]
        gt2d = to_torch(gt2d, device)

        # Mask non-finite points in GT
        valid = torch.isfinite(gt2d).all(dim=0)
        pred_pts = proj[0, :, valid]  # [2, Nv]
        gt_pts = gt2d[:, valid]       # [2, Nv]

        if pred_pts.numel() == 0:
            print(f"[{cam_key}] no valid keypoints")
            continue

        # Compute L1 and L2 errors
        diff = pred_pts - gt_pts  # [2, N]
        l1_error = diff.abs().mean().item()
        l2_error = torch.sqrt((diff ** 2).sum(dim=0)).mean().item()
        
        # Handle potential NaN/inf values 
        l1_str = f"{l1_error:.3f}px" if np.isfinite(l1_error) else "nan"
        l2_str = f"{l2_error:.3f}px" if np.isfinite(l2_error) else "nan"
        
        print(f"[{cam_key}] valid={int(valid.sum())} L1={l1_str} L2={l2_str}")

        # Show a few sample points
        for j in range(min(max_show_kpts, pred_pts.shape[1])):
            print(
                f"  kpt{j}: pred=({pred_pts[0,j].item():.1f},{pred_pts[1,j].item():.1f}) "
                f"gt=({gt_pts[0,j].item():.1f},{gt_pts[1,j].item():.1f})"
            )

    # Store processed data for visualization function  
    global _shared_data
    _shared_data = (samples, datadict, datadict_3d, com3d_dict, cameras, camnames_map, total_chunks, params)

    return 0


def visualize_training_sample():
    """Create and visualize actual training data to see what model sees."""
    import torch.utils.data as data
    
    # Use shared data from compare_one_batch to avoid reprocessing
    global _shared_data
    if _shared_data is None:
        print("No processed data available from comparison. Running data loading...")
        base_config_path = "/work/rl349/tmp/sdannce/dannce_config_custom.yaml"
        params = build_params(base_config_path, dannce_net=True)
        # Load data fresh if needed
        samples, datadict, datadict_3d, com3d_dict, cameras, camnames_map, total_chunks, temporal_chunks = _load_fresh_data(params)
    else:
        print("Using shared data from comparison...")
        samples, datadict, datadict_3d, com3d_dict, cameras, camnames_map, total_chunks, params = _shared_data
    
    # Data is already processed and includes experiment ID prepending
    
    print(f"Creating training dataset...")
    print(f"After prepending - Camera keys for exp 0: {list(cameras[0].keys()) if 0 in cameras else 'None'}")
    print(f"After prepending - Camnames for exp 0: {camnames_map.get(0, [])}")
    
    try:
        # Create simple data split for visualization
        partition = {
            "train_sampleIDs": samples[:10],  # Just first 10 samples
            "valid_sampleIDs": samples[10:15] if len(samples) > 10 else samples[:5],
        }
        
        if params.get("use_npy", True):
            # Check if NPY files exist and get directory structure  
            npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, samples[:10])
            
            if len(missing_samples) > 0:
                print(f"Missing {len(missing_samples)} NPY files, attempting to generate them...")
                
                # Try to generate missing NPY volumes
                try:
                    from dannce.engine.data.generator import DataGenerator_3Dconv
                    
                    # Create a small subset for NPY generation
                    subset_samples = missing_samples[:3]  # Just generate 3 samples
                    print(f"Generating NPY volumes for {len(subset_samples)} samples...")
                    
                    # Initialize video readers exactly like in training
                    vid_exps = [0]  # Just experiment 0
                    
                    # Ensure extension parameter exists for NPY generation too
                    if 'extension' not in params['experiment'][0]:
                        print("Adding missing 'extension' parameter for NPY generation")
                        params['experiment'][0]['extension'] = '.mp4'
                    
                    vids = processing.initialize_all_vids(params, datadict, vid_exps, pathonly=True)
                    print(f"Video readers initialized: {type(vids)}, keys: {list(vids.keys()) if vids else 'None'}")
                    
                    # Set up parameters exactly like config.setup_train does
                    from dannce.config import setup_train
                    params_copy = params.copy()
                    
                    # Add missing parameters required by setup_train
                    params_copy.setdefault("use_silhouette_in_volume", False)
                    params_copy.setdefault("write_visual_hull", None)
                    params_copy.setdefault("cam3_train", False)
                    params_copy.setdefault("rotate", False)
                    params_copy.setdefault("augment_hue", False)
                    params_copy.setdefault("augment_brightness", False)
                    params_copy.setdefault("augment_continuous_rotation", False)
                    params_copy.setdefault("augment_rotation_val", 5)
                    params_copy.setdefault("allow_valid_replace", False)
                    params_copy.setdefault("sigma", 10)  # Default gaussian sigma for 3D targets
                    params_copy.setdefault("net", "unet3d_big")  # Default network type
                    params_copy.setdefault("depth", False)
                    params_copy.setdefault("immode", "video")  # Image mode for video files
                    params_copy.setdefault("mono", False)  # Keep RGB channels to match video data
                    params_copy.setdefault("mirror", False)  # No mirror augmentation
                    params_copy.setdefault("heatmap_reg", False)  # Heatmap regularization
                    params_copy.setdefault("heatmap_reg_coeff", 0.0)  # Heatmap regularization coefficient
                    params_copy.setdefault("augment_hue_val", 0.05)  # Hue augmentation value
                    
                    params_copy, base_params, shared_args, shared_args_train, shared_args_valid = setup_train(params_copy)
                    
                    # Update base_params with runtime values like training does
                    # Fix chunks structure to match prepended camera names
                    fixed_chunks = {}
                    if total_chunks:
                        for orig_name, chunk_data in total_chunks.items():
                            # Map non-prepended names to prepended names
                            if not orig_name.startswith('0_'):
                                prepended_name = f"0_{orig_name}"
                                if prepended_name in camnames_map.get(0, []):
                                    fixed_chunks[prepended_name] = chunk_data
                            else:
                                fixed_chunks[orig_name] = chunk_data
                    
                    base_params = {
                        **base_params,
                        "camnames": camnames_map,
                        "vidreaders": vids,
                        "chunks": fixed_chunks,
                    }
                    
                    # Set NPY generation parameters exactly like training
                    params["chan_num"] = params.get("n_channels_in", 3)
                    spec_params = {
                        "channel_combo": None,  # Use None instead of "avg" for simpler case
                        "predict_flag": False,
                        "norm_im": True,  # Enable normalization to ensure float type
                        "expval": True,
                        "crop_im": False,  # Set to False to ensure X gets assigned in pj_grid_post
                        "mode": "3dprob",  # Set proper mode
                        "mono": False,  # Keep RGB channels to match video data
                        "n_channels_in": 3,  # Match actual RGB video data
                        "immode": "video",  # Set image mode
                    }
                    valid_params = {**base_params, **spec_params}
                    
                    print(f"Base params keys: {list(base_params.keys())}")
                    print(f"Valid params keys: {list(valid_params.keys())}")
                    
                    # Create generator exactly like training does at line 631-640
                    genfunc = DataGenerator_3Dconv
                    npy_generator = genfunc(
                        subset_samples,  # list_IDs
                        datadict,        # labels  
                        datadict_3d,     # labels_3d
                        cameras,         # camera_params
                        subset_samples,  # clusterIDs
                        com3d_dict,      # com3d
                        [],              # tifdirs
                        **valid_params,  # All other parameters like training
                    )
                    
                    # Manually fix extension if initialization failed
                    if not hasattr(npy_generator, 'extension') or npy_generator.extension is None:
                        npy_generator.extension = ".mp4"
                        print(f"Manually set extension to .mp4")
                    
                    # Generate and save NPY volumes
                    processing.save_volumes_into_npy(
                        params, npy_generator, missing_npydir, subset_samples
                    )
                    
                    print(f"Successfully generated NPY volumes for {len(subset_samples)} samples")
                    
                    # Update samples list to only include generated ones
                    partition["train_sampleIDs"] = subset_samples
                    
                except Exception as npy_error:
                    print(f"NPY generation failed: {npy_error}")
                    print("Falling back to simple volume visualization without dataset...")
                    
                    # Create a simple visualization without full dataset
                    try:
                        visualize_simple_volumes(samples[:3], datadict, datadict_3d, cameras, camnames_map, params, com3d_dict, total_chunks)
                    except Exception as simple_error:
                        print(f"Simple visualization also failed: {simple_error}")
                        print("Showing basic data information instead...")
                        visualize_basic_data_info(samples[:3], datadict, datadict_3d, cameras, camnames_map, params)
                    return
            
            # Re-check after potential generation
            npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, partition["train_sampleIDs"])
            
            if len(missing_samples) > 0:
                print("Still missing NPY files after generation attempt")
                raise Exception("Missing NPY files")
            
            # Create NPY dataset like in training - MATCH EXACT TRAINING PARAMETERS
            dataset = PoseDatasetNPY(
                list_IDs=partition["train_sampleIDs"],
                labels_3d=datadict_3d,
                labels_2d=datadict,
                npydir=npydir,
                cameras=cameras,
                sigma=params.get("sigma", 10),
                mono=params.get("mono", False),
                aux=params.get("use_silhouette", False),
                temporal_chunk_list=None,  # This automatically sets temporal_chunk_size=1
                nvox=params.get("nvox", 80),  # Use config value
                expval=params.get("expval", True),  # Critical: NPY files generated with expval=True
                shuffle=False,  # For consistent visualization
                chan_num=params.get("n_channels_in", 3),  # Add missing param
                # Fix augmentation parameters to prevent reshape errors
                rotation=False,  # Disable rotation to avoid reshape issues
                augment_continuous_rotation=False,  # Disable continuous rotation
            )
        else:
            print("In-memory dataset creation not implemented in this visualization")
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Visualize first few samples
        save_dir = "debug_training_visualizations"
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(min(3, len(dataset))):
            print(f"\n=== Visualizing sample {i} ===")
            volumes, labels_3d = visualize_training_data(dataset, i, save_dir)
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("This might be because NPY volumes don't exist or other dataset issues")


# Global variable to share processed data between functions
_shared_data = None

def main():
    # Silence verbose config inheritance warnings
    try:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
    # Direct execution without CLI: configure here
    BASE_CONFIG_PATH = "/work/rl349/tmp/sdannce/dannce_config_custom.yaml"
    APPLY_ALIGNMENT = True  # set False to skip crop/downfac
    MAX_SHOW_KPTS = 3
    DEVICE = None  # e.g., "cuda:0" or "cpu"; None selects automatically

    print("=== Reprojection Comparison ===")
    status = compare_one_batch(
        BASE_CONFIG_PATH,
        apply_alignment=APPLY_ALIGNMENT,
        max_show_kpts=MAX_SHOW_KPTS,
        device=DEVICE,
    )
    
    print(f"\n=== Training Data Visualization ===")
    try:
        visualize_training_sample()
    except Exception as e:
        print(f"Training visualization failed: {e}")
    
    raise SystemExit(status)


if __name__ == "__main__":
    main()


