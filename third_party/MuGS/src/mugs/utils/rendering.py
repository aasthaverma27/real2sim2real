"""
MuGS Rendering Utilities

Common functions for hybrid rendering pipeline:
- PLY loading
- MuJoCo rendering
- 3DGS rendering
- Hybrid compositing

Author: MuGS Team
Date: 2026-05-02
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np


def load_ply_gaussians(ply_path: Path) -> Dict[str, np.ndarray]:
    """
    Load 3D Gaussian Splatting parameters from PLY file.

    Args:
        ply_path: Path to PLY file

    Returns:
        Dictionary with keys: means, scales, quats, opacities, sh_coeffs
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    # Extract positions
    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)

    # Extract scales (log-space in PLY)
    scales = np.stack(
        [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=-1
    ).astype(np.float32)
    scales = np.exp(scales)  # Convert from log-space

    # Extract quaternions (wxyz format in PLY)
    quats = np.stack(
        [
            vertices["rot_0"],  # w
            vertices["rot_1"],  # x
            vertices["rot_2"],  # y
            vertices["rot_3"],  # z
        ],
        axis=-1,
    ).astype(np.float32)

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

    # Extract opacities (sigmoid-space in PLY)
    opacities = vertices["opacity"].astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities))  # Sigmoid

    # Extract spherical harmonics coefficients
    sh_keys = [
        key
        for key in vertices.data.dtype.names
        if key.startswith("f_dc_") or key.startswith("f_rest_")
    ]
    sh_keys = sorted(sh_keys)
    sh_coeffs = np.stack([vertices[key] for key in sh_keys], axis=-1).astype(np.float32)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }


def render_mujoco_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    renderer: Optional[mujoco.Renderer] = None,
) -> np.ndarray:
    """
    Render RGB image from MuJoCo.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera
        width: Image width
        height: Image height
        renderer: Optional pre-created renderer

    Returns:
        RGB image (H, W, 3) in range [0, 1]
    """
    if renderer is None:
        renderer = mujoco.Renderer(model, height=height, width=width)

    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    return rgb.astype(np.float32) / 255.0


def render_mujoco_segmentation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    renderer: Optional[mujoco.Renderer] = None,
) -> np.ndarray:
    """
    Render segmentation IDs from MuJoCo.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera
        width: Image width
        height: Image height
        renderer: Optional pre-created renderer

    Returns:
        Segmentation IDs (H, W) with geom IDs
    """
    if renderer is None:
        renderer = mujoco.Renderer(model, height=height, width=width)

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    # Extract geom IDs from red channel
    seg_ids = seg[:, :, 0].astype(np.int32)

    return seg_ids


def create_robot_mask(
    seg_ids: np.ndarray, model: mujoco.MjModel, robot_geom_names: list
) -> np.ndarray:
    """
    Extract robot mask from segmentation IDs.

    Args:
        seg_ids: Segmentation IDs (H, W)
        model: MuJoCo model
        robot_geom_names: List of robot geom names

    Returns:
        Binary mask (H, W) where 1=robot, 0=background
    """
    robot_geom_ids = []
    for name in robot_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            robot_geom_ids.append(geom_id)

    # Create binary mask
    mask = np.zeros_like(seg_ids, dtype=np.uint8)
    for geom_id in robot_geom_ids:
        mask[seg_ids == geom_id] = 1

    return mask


def composite_images(
    mujoco_rgb: np.ndarray, gs_rgb: np.ndarray, robot_mask: np.ndarray
) -> np.ndarray:
    """
    Composite MuJoCo and 3DGS images using robot mask.

    Args:
        mujoco_rgb: MuJoCo RGB (H, W, 3)
        gs_rgb: 3DGS RGB (H, W, 3)
        robot_mask: Binary mask (H, W)

    Returns:
        Composited RGB image (H, W, 3)
    """
    mask_3ch = robot_mask[..., None].astype(np.float32)
    composite = mujoco_rgb * mask_3ch + gs_rgb * (1 - mask_3ch)

    return composite


def extract_mujoco_camera_params(
    model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, width: int, height: int
) -> Dict[str, np.ndarray]:
    """
    Extract camera parameters from MuJoCo for 3DGS rendering.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera
        width: Image width
        height: Image height

    Returns:
        Dictionary with keys: position, lookat, up, fov
    """
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # Get camera pose in world frame
    camera_pos = data.cam_xpos[camera_id].copy()
    camera_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()

    # Camera looks along -Z axis in camera frame
    camera_forward = -camera_mat[:, 2]
    camera_up = camera_mat[:, 1]
    camera_lookat = camera_pos + camera_forward

    # Get FOV
    fov = model.cam_fovy[camera_id]

    return {"position": camera_pos, "lookat": camera_lookat, "up": camera_up, "fov": fov}
