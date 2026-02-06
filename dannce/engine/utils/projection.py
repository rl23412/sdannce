import mat73
import numpy as np
import scipy.io as sio
import torch


# helper functions
def project_to_2d_single_camera(
    pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3d points to 2d for a single camera.
    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    M = np.concatenate((R, t), axis=0) @ K
    projPts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ M
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:3]  # Use 2:3 instead of 2: for proper broadcasting

    return projPts





# Alias for compatibility with existing DANNCE code
project_to_2d = project_to_2d_single_camera


def distortPoints(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.
    Ported from Matlab 2018a
    """
    # unpack the intrinisc matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis, :]

    # normalize the points
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = np.stack((xNorm, yNorm)).T
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * np.array([alpha, alpha]).T
        + np.stack((dxTangential, dyTangential)).T
    )

    # # convert back to pixels
    distortedPointsX = (
        (distortedNormalizedPoints[:, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, 1])
    )
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy
    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def project_to_2d_torch(points_3d, K, R, t):
    """
    Differentiable 3D->2D projection using torch - optimized for batched operations.
    
    This function performs batched projection operations to avoid gradient issues
    with sequential processing while maintaining numerical equivalence to numpy.

    Args:
        points_3d (torch.Tensor): [B, 3, N] 3D points.
        K (torch.Tensor): [3, 3] intrinsic matrix.
        R (torch.Tensor): [3, 3] rotation matrix.
        t (torch.Tensor): [1, 3] or [3] translation vector (row-wise, Matlab convention).

    Returns:
        torch.Tensor: [B, 2, N] projected 2D points in pixel coordinates.
    """
    assert points_3d.dim() == 3 and points_3d.shape[1] == 3, "points_3d must be [B, 3, N]"

    device = points_3d.device
    dtype = points_3d.dtype

    # Convert inputs to proper device/dtype
    K = K.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)

    # Handle translation vector format - ensure [1, 3]
    if t.dim() == 1:
        t = t.view(1, 3)
    elif t.shape == (3, 1):
        t = t.view(1, 3)

    # Efficient batched processing
    B, _, N = points_3d.shape
    
    # Construct projection matrix M = [R; t] @ K
    M = torch.cat((R, t), dim=0) @ K  # [4, 3]
    
    # Reshape points for batched matrix multiplication: [B, N, 3]
    pts_reshaped = points_3d.permute(0, 2, 1)  # [B, N, 3]
    
    # Add homogeneous coordinate: [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=dtype)
    pts_homo = torch.cat([pts_reshaped, ones], dim=2)  # [B, N, 4]
    
    # Batched projection: [B, N, 4] @ [4, 3] -> [B, N, 3]
    projPts = pts_homo @ M  # [B, N, 3]
    
    # Perspective division (avoiding in-place operations)
    xy_coords = projPts[:, :, :2]  # [B, N, 2]
    z_coords = projPts[:, :, 2:3]  # [B, N, 1]
    normalized_coords = xy_coords / z_coords  # [B, N, 2]
    
    # Convert to [B, 2, N] format
    result = normalized_coords.permute(0, 2, 1)  # [B, 2, N]
    
    return result


def distort_points_torch(points_2d, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """
    Differentiable distortion using torch - optimized for batched operations.
    
    This function performs batched distortion operations while maintaining identical
    NaN handling behavior to the numpy version. Distortion is applied regardless
    of NaN values, matching numpy's behavior.

    Args:
        points_2d (torch.Tensor): [B, 2, N] 2D pixel coordinates.
        intrinsicMatrix (torch.Tensor): [3, 3] camera intrinsics.
        radialDistortion (torch.Tensor): [2] or [3] radial coefficients
        tangentialDistortion (torch.Tensor): [2] tangential coefficients

    Returns:
        torch.Tensor: [B, 2, N] distorted 2D pixel coordinates.
    """
    assert points_2d.dim() == 3 and points_2d.shape[1] == 2, "points_2d must be [B, 2, N]"

    device = points_2d.device
    dtype = points_2d.dtype
    
    # Convert to proper device/dtype
    intrinsicMatrix = intrinsicMatrix.to(device=device, dtype=dtype)
    radialDistortion = radialDistortion.to(device=device, dtype=dtype)
    tangentialDistortion = tangentialDistortion.to(device=device, dtype=dtype)

    B, _, N = points_2d.shape
    
    # Unpack the intrinsic matrix (same for all batch elements)
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1] 
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # Convert [B, 2, N] -> [B, N, 2] for easier processing
    points = points_2d.permute(0, 2, 1)  # [B, N, 2]
    
    # Center the points (batched)
    center = torch.tensor([cx, cy], device=device, dtype=dtype)
    centeredPoints = points - center[None, None, :]  # [B, N, 2]

    # Normalize the points (batched)
    yNorm = centeredPoints[:, :, 1] / fy  # [B, N]
    xNorm = (centeredPoints[:, :, 0] - skew * yNorm) / fx  # [B, N]

    # Compute radial distortion (batched)
    r2 = xNorm ** 2 + yNorm ** 2  # [B, N]
    r4 = r2 * r2
    r6 = r2 * r4

    # Handle radial distortion coefficients
    k = torch.zeros(3, device=device, dtype=dtype)
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6  # [B, N]

    # Compute tangential distortion (batched)
    p = tangentialDistortion
    xyProduct = xNorm * yNorm  # [B, N]
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)  # [B, N]
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct  # [B, N]

    # Apply the distortion to the points (batched)
    normalizedPoints = torch.stack((xNorm, yNorm), dim=2)  # [B, N, 2]
    alpha_expanded = alpha.unsqueeze(2).expand(-1, -1, 2)  # [B, N, 2]
    tangential_distortion = torch.stack((dxTangential, dyTangential), dim=2)  # [B, N, 2]
    
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * alpha_expanded
        + tangential_distortion
    )  # [B, N, 2]

    # Convert back to pixels (batched)
    distortedPointsX = (
        (distortedNormalizedPoints[:, :, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, :, 1])
    )  # [B, N]
    distortedPointsY = distortedNormalizedPoints[:, :, 1] * fy + cy  # [B, N]
    
    # Stack and convert back to [B, 2, N] format
    distortedPoints = torch.stack((distortedPointsX, distortedPointsY), dim=2)  # [B, N, 2]
    result = distortedPoints.permute(0, 2, 1)  # [B, 2, N]
    
    return result


def load_cameras(path):
    mat73_flag = False
    try:
        d = sio.loadmat(path)
        camnames = [cam[0] for cam in d["camnames"][0]]
    except:
        d = mat73.loadmat(path)
        camnames = [name[0] for name in d["camnames"]]
        mat73_flag = True
    fns = ["K", "RDistort", "TDistort", "r", "t"]

    cam_params = d["params"]
    cameras = {}
    for i, camname in enumerate(camnames):
        cameras[camname] = {}
        for j, fn in enumerate(fns):
            if mat73_flag:
                cameras[camname][fn] = cam_params[i][0][fn]
            else:
                cameras[camname][fn] = cam_params[i][0][0][0][j]
        if len(cameras[camname]["t"].shape) == 1:
            cameras[camname]["t"] = cameras[camname]["t"][np.newaxis, ...]
    return cameras
