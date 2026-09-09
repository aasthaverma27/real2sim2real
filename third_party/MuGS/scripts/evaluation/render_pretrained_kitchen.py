"""
Render pretrained Mip-NeRF 360 kitchen scene using gsplat.

Loads the official INRIA pretrained kitchen model and renders test views.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization


def load_official_ply(ply_path: Path):
    """
    Load official 3DGS PLY format with spherical harmonics.

    Returns dict with numpy arrays (CPU).
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Spherical harmonics - convert DC coefficients to RGB
    # SH DC coefficients are stored in f_dc_0, f_dc_1, f_dc_2
    # RGB = 0.5 + SH_C0 * sh_dc (where SH_C0 = 0.28209479177387814)
    SH_C0 = 0.28209479177387814
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
    colors = 0.5 + SH_C0 * sh_dc
    colors = np.clip(colors, 0, 1)  # Clamp to [0, 1]

    # Opacity (stored as logit, need to apply sigmoid)
    opacity_logit = vertex['opacity']
    opacities = 1 / (1 + np.exp(-opacity_logit))  # sigmoid

    # Scales (stored as log, need to apply exp)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    scales = np.exp(scales)

    # Rotations (quaternions in w, x, y, z order)
    quats = np.stack([
        vertex['rot_0'],  # w
        vertex['rot_1'],  # x
        vertex['rot_2'],  # y
        vertex['rot_3']   # z
    ], axis=1)

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    return {
        'means': positions,
        'colors': colors,
        'opacities': opacities,  # (N,)
        'scales': scales,
        'quats': quats,
    }


def load_cameras(cameras_json_path: Path):
    """Load camera parameters from cameras.json."""
    with open(cameras_json_path) as f:
        cameras = json.load(f)
    return cameras


def camera_to_view_matrix(position, rotation):
    """Convert camera position and rotation to view matrix."""
    R = np.array(rotation).T  # Transpose to get world-to-camera
    t = -R @ np.array(position)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t

    return view_matrix


def render_camera(gaussians, camera, width=640, height=480, device='cuda'):
    """Render a single camera view."""

    # Camera intrinsics
    fx = camera['fx'] * (width / camera['width'])
    fy = camera['fy'] * (height / camera['height'])
    cx = width / 2
    cy = height / 2

    # View matrix
    view_matrix = camera_to_view_matrix(camera['position'], camera['rotation'])
    viewmat = torch.from_numpy(view_matrix).float().to(device)

    # Projection matrix (perspective)
    near = 0.01
    far = 100.0
    proj_matrix = torch.zeros(4, 4, device=device)
    proj_matrix[0, 0] = 2 * fx / width
    proj_matrix[1, 1] = 2 * fy / height
    proj_matrix[0, 2] = 2 * (cx / width) - 1
    proj_matrix[1, 2] = 2 * (cy / height) - 1
    proj_matrix[2, 2] = far / (far - near)
    proj_matrix[2, 3] = -(far * near) / (far - near)
    proj_matrix[3, 2] = 1.0

    # Render with gsplat
    render_colors, render_alphas, info = rasterization(
        means=gaussians['means'],
        quats=gaussians['quats'],
        scales=gaussians['scales'],
        opacities=gaussians['opacities'],
        colors=gaussians['colors'],
        viewmats=viewmat[None],
        Ks=torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device)[None],
        width=width,
        height=height,
        packed=False,
    )

    # Convert to RGB image
    rgb = render_colors[0].clamp(0, 1)
    rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)

    return rgb


def main():
    """Render pretrained kitchen scene."""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"
    output_dir = base_dir / "outputs/pretrained_kitchen"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained kitchen model...")
    print(f"  Model: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1e6:.1f} MB")

    # Load Gaussians (official 3DGS format)
    gaussians_np = load_official_ply(model_path)

    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = {
        'means': torch.from_numpy(gaussians_np['means']).float().to(device),
        'quats': torch.from_numpy(gaussians_np['quats']).float().to(device),
        'scales': torch.from_numpy(gaussians_np['scales']).float().to(device),
        'opacities': torch.from_numpy(gaussians_np['opacities']).float().to(device),
        'colors': torch.from_numpy(gaussians_np['colors']).float().to(device),
    }

    n_gaussians = len(gaussians['means'])
    print(f"  Loaded {n_gaussians:,} Gaussians")

    # Load cameras
    cameras = load_cameras(cameras_path)
    print(f"  Found {len(cameras)} camera views")

    # Render a few test views
    render_width = 960
    render_height = 640

    test_indices = [0, 50, 100, 150, 200]  # Sample cameras

    print(f"\nRendering {len(test_indices)} test views at {render_width}x{render_height}...")

    for idx in test_indices:
        if idx >= len(cameras):
            continue

        camera = cameras[idx]
        print(f"  Rendering camera {idx} ({camera['img_name']})...", end=' ')

        rgb = render_camera(gaussians, camera, render_width, render_height, device)

        # Save image
        output_path = output_dir / f"camera_{idx:03d}_{camera['img_name']}.jpg"
        Image.fromarray(rgb).save(output_path, quality=95)

        print(f"✓ {output_path.name}")

    print(f"\n✅ Rendered images saved to: {output_dir}")
    print(f"\nTo view all images: ls {output_dir}/*.jpg")


if __name__ == "__main__":
    main()
