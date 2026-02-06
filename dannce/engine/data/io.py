"""Label3D data loading and saving operations."""
from typing import Dict, List, Text, Union

import mat73
import numpy as np
import scipy.io as sio


def load_label3d_data(path: Text, key: Text):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
    try:
        d = sio.loadmat(path)[key]
        dataset = [f[0] for f in d]

        # Data are loaded in this annoying structure where the array
        # we want is at dataset[i][key][0,0], as a nested array of arrays.
        # Simplify this structure (a numpy record array) here.
        # Additionally, cannot use views here because of shape mismatches. Define
        # new dict and return.
        data = []
        for d in dataset:
            d_ = {}
            for key in d.dtype.names:
                d_[key] = d[key][0, 0]
            data.append(d_)
    except:
        d = mat73.loadmat(path)[key]
        data = [f[0] for f in d]
    return data


def load_camera_params(path: Text) -> List[Dict]:
    """Load camera parameters from Label3D file.

    Args:
        path (Text): Path to Label3D file

    Returns:
        List[Dict]: List of camera parameter dictionaries.
    """
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
        if len(p["t"].shape) == 1:
            p["t"] = p["t"][np.newaxis, ...]
    return params


def load_sync(path: Text) -> List[Dict]:
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path: Text) -> List[Dict]:
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path: Text) -> Dict:
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    try:
        d = sio.loadmat(path)["com"]
        data = {}
        data["com3d"] = d["com3d"][0, 0]
        data["sampleID"] = d["sampleID"][0, 0].astype(int)
    except:
        data = mat73.loadmat(path)["com"]
        data["sampleID"] = data["sampleID"].astype(int)
    return data


def load_label2d_confidence(path: Text) -> Union[List[Dict], None]:
    """Load 2D confidence scores from Label3D file.
    
    Label3D files may contain confidence scores for 2D keypoint annotations.
    These scores indicate the reliability of each keypoint annotation.
    
    Args:
        path (Text): Path to Label3D file
        
    Returns:
        Union[List[Dict], None]: List of confidence data dictionaries per experiment,
                                or None if no confidence data available
    """
    try:
        # Try to load confidence data from Label3D file
        confidence_data = load_label3d_data(path, "confidence")
        
        # Process confidence data structure similar to labelData
        processed_confidence = []
        for conf_exp in confidence_data:
            conf_dict = {}
            # Extract confidence matrices per camera and keypoint
            for key in conf_exp.dtype.names if hasattr(conf_exp, 'dtype') else conf_exp.keys():
                if key in conf_exp:
                    conf_dict[key] = conf_exp[key]
            processed_confidence.append(conf_dict)
            
        return processed_confidence
        
    except (KeyError, TypeError):
        # No confidence data available - this is normal for many datasets
        return None
    except Exception as e:
        print(f"Warning: Could not load confidence data from {path}: {e}")
        return None


def create_default_confidence(labelData: List[Dict], default_confidence: float = 1.0) -> List[Dict]:
    """Create default confidence scores when not available in Label3D file.
    
    Args:
        labelData: List of labelData dictionaries from Label3D
        default_confidence: Default confidence value to assign (1.0 = full confidence)
        
    Returns:
        List[Dict]: List of confidence dictionaries matching labelData structure
    """
    confidence_data = []
    
    for exp_data in labelData:
        conf_dict = {}
        
        # Create confidence scores matching the labelData structure
        if 'data_2d' in exp_data and exp_data['data_2d'] is not None:
            data_2d = exp_data['data_2d']
            if isinstance(data_2d, np.ndarray) and len(data_2d.shape) >= 3:
                # data_2d shape: (n_frames, n_cameras, n_keypoints, 2)
                # Create confidence: (n_frames, n_cameras, n_keypoints)
                n_frames, n_cameras, n_keypoints = data_2d.shape[:3]
                conf_dict['data_2d_confidence'] = np.full(
                    (n_frames, n_cameras, n_keypoints), 
                    default_confidence, 
                    dtype=np.float32
                )
        
        confidence_data.append(conf_dict)
    
    return confidence_data


def load_camnames(path: Text) -> Union[List, None]:
    """Load camera names from .mat file.

    Args:
        path (Text): Path to .mat file with "camnames" field

    Returns:
        Union[List, None]: List of cameranames
    """
    try:
        label_3d_file = sio.loadmat(path)
        if "camnames" in label_3d_file:
            names = label_3d_file["camnames"][:]
            if len(names) != len(label_3d_file["labelData"]):
                camnames = [name[0] for name in names[0]]
            else:
                camnames = [name[0][0] for name in names]
        else:
            camnames = None
    except:
        label_3d_file = mat73.loadmat(path)
        if "camnames" in label_3d_file:
            camnames = [name[0] for name in label_3d_file["camnames"]]
    return camnames
