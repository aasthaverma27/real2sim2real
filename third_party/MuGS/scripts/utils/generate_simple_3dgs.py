#!/usr/bin/env python3
"""
Generate simple 3DGS PLY files for testing

Creates synthetic 3D Gaussian Splatting models programmatically.
Useful for testing gsplat rendering without needing real captured assets.

Author: MuGS Team
Date: 2026-05-02
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def create_sphere_gaussians(
    center: np.ndarray,
    radius: float,
    num_gaussians: int = 500,
    color: np.ndarray = None,
) -> dict:
    """
    Create Gaussians arranged in a sphere.

    Args:
        center: (3,) center position
        radius: sphere radius
        num_gaussians: number of Gaussians
        color: (3,) RGB color [0-1], default random

    Returns:
        Dict with gaussian parameters
    """
    # Random points on sphere surface using Fibonacci sphere
    indices = np.arange(num_gaussians)
    phi = np.arccos(1 - 2 * (indices + 0.5) / num_gaussians)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    means = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Small scale for smooth appearance
    scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.01

    # Random rotations (quaternions: w, x, y, z)
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # Identity rotation

    # Opacities (all visible)
    opacities = np.ones((num_gaussians, 1), dtype=np.float32) * 0.9

    # Colors (SH coefficients, DC component only for simplicity)
    if color is None:
        color = np.random.rand(3)

    # SH DC component: RGB to SH
    # SH_C0 = 0.28209479177387814
    sh_dc = np.zeros((num_gaussians, 3), dtype=np.float32)
    sh_dc[:] = (color - 0.5) / 0.28209479177387814

    # No higher-order SH (set to zeros)
    sh_rest = np.zeros((num_gaussians, 45), dtype=np.float32)  # 15 coeffs * 3 channels

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'sh_dc': sh_dc,
        'sh_rest': sh_rest,
    }


def create_cylinder_gaussians(
    center: np.ndarray,
    radius: float,
    height: float,
    num_gaussians: int = 500,
    color: np.ndarray = None,
) -> dict:
    """Create Gaussians arranged in a cylinder (like a mug body)."""
    # Points on cylinder surface
    num_height = int(np.sqrt(num_gaussians))
    num_circle = num_gaussians // num_height

    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    z = np.linspace(-height/2, height/2, num_height)

    theta_grid, z_grid = np.meshgrid(theta, z)
    theta_flat = theta_grid.flatten()
    z_flat = z_grid.flatten()

    # Take exactly num_gaussians points
    actual_num = len(theta_flat)
    if actual_num > num_gaussians:
        theta_flat = theta_flat[:num_gaussians]
        z_flat = z_flat[:num_gaussians]
        actual_num = num_gaussians

    num_gaussians = actual_num  # Update to actual count

    x = radius * np.cos(theta_flat) + center[0]
    y = radius * np.sin(theta_flat) + center[1]
    z = z_flat + center[2]

    means = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Small scale
    scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.008

    # Identity rotations
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Opacities
    opacities = np.ones((num_gaussians, 1), dtype=np.float32) * 0.95

    # Colors
    if color is None:
        color = np.array([0.3, 0.6, 0.8])  # Blue for mug

    sh_dc = np.zeros((num_gaussians, 3), dtype=np.float32)
    sh_dc[:] = (color - 0.5) / 0.28209479177387814
    sh_rest = np.zeros((num_gaussians, 45), dtype=np.float32)

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'sh_dc': sh_dc,
        'sh_rest': sh_rest,
    }


def save_gaussians_to_ply(gaussians: dict, output_path: Path):
    """
    Save Gaussian parameters to PLY file.

    PLY format for 3DGS:
    - x, y, z: mean positions
    - nx, ny, nz: normals (unused, set to 0)
    - f_dc_0, f_dc_1, f_dc_2: DC SH coefficients (RGB)
    - f_rest_0 ... f_rest_44: higher-order SH coefficients
    - opacity: opacity
    - scale_0, scale_1, scale_2: scale
    - rot_0, rot_1, rot_2, rot_3: rotation quaternion (w, x, y, z)
    """
    num_gaussians = len(gaussians['means'])

    # Create structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # Add f_rest_0 to f_rest_44
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))

    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    vertex_data = np.zeros(num_gaussians, dtype=dtype)

    # Fill positions
    vertex_data['x'] = gaussians['means'][:, 0]
    vertex_data['y'] = gaussians['means'][:, 1]
    vertex_data['z'] = gaussians['means'][:, 2]

    # Normals (unused)
    vertex_data['nx'] = 0
    vertex_data['ny'] = 0
    vertex_data['nz'] = 0

    # SH DC
    vertex_data['f_dc_0'] = gaussians['sh_dc'][:, 0]
    vertex_data['f_dc_1'] = gaussians['sh_dc'][:, 1]
    vertex_data['f_dc_2'] = gaussians['sh_dc'][:, 2]

    # SH rest
    for i in range(45):
        vertex_data[f'f_rest_{i}'] = gaussians['sh_rest'][:, i]

    # Opacity
    vertex_data['opacity'] = gaussians['opacities'][:, 0]

    # Scales
    vertex_data['scale_0'] = gaussians['scales'][:, 0]
    vertex_data['scale_1'] = gaussians['scales'][:, 1]
    vertex_data['scale_2'] = gaussians['scales'][:, 2]

    # Rotations (quaternion: w, x, y, z)
    vertex_data['rot_0'] = gaussians['quats'][:, 0]
    vertex_data['rot_1'] = gaussians['quats'][:, 1]
    vertex_data['rot_2'] = gaussians['quats'][:, 2]
    vertex_data['rot_3'] = gaussians['quats'][:, 3]

    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    # Write PLY file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([vertex_element]).write(output_path)

    print(f"✅ Saved {num_gaussians} Gaussians to: {output_path}")


def main():
    """Generate test 3DGS assets."""
    print("="*60)
    print("Generate Simple 3DGS Test Assets")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent
    assets_dir = project_root / "assets" / "objects"

    # 1. Blue mug (cylinder)
    print("\n1. Generating blue mug...")
    mug_gaussians = create_cylinder_gaussians(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.035,
        height=0.12,
        num_gaussians=800,
        color=np.array([0.3, 0.6, 0.8])  # Blue
    )
    save_gaussians_to_ply(mug_gaussians, assets_dir / "kitchen" / "mug_blue.ply")

    # 2. White plate (flat cylinder)
    print("\n2. Generating white plate...")
    plate_gaussians = create_cylinder_gaussians(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.08,
        height=0.02,
        num_gaussians=600,
        color=np.array([0.95, 0.95, 0.95])  # White
    )
    save_gaussians_to_ply(plate_gaussians, assets_dir / "kitchen" / "plate_white.ply")

    # 3. Red ball (sphere)
    print("\n3. Generating red ball...")
    ball_gaussians = create_sphere_gaussians(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.04,
        num_gaussians=500,
        color=np.array([0.8, 0.2, 0.2])  # Red
    )
    save_gaussians_to_ply(ball_gaussians, assets_dir / "misc" / "ball_red.ply")

    # 4. Green bowl (sphere with top cut off)
    print("\n4. Generating green bowl...")
    bowl_gaussians = create_sphere_gaussians(
        center=np.array([0.0, 0.0, -0.02]),  # Shift down
        radius=0.06,
        num_gaussians=700,
        color=np.array([0.4, 0.7, 0.3])  # Green
    )
    # Filter to keep only bottom half
    mask = bowl_gaussians['means'][:, 2] < 0.0
    for key in bowl_gaussians:
        bowl_gaussians[key] = bowl_gaussians[key][mask]

    save_gaussians_to_ply(bowl_gaussians, assets_dir / "kitchen" / "bowl_green.ply")

    print("\n" + "="*60)
    print("✅ Generated 4 test 3DGS assets")
    print("="*60)
    print(f"\n📁 Assets location: {assets_dir}")
    print("\nGenerated files:")
    for ply_file in sorted(assets_dir.rglob("*.ply")):
        size_kb = ply_file.stat().st_size / 1024
        print(f"   - {ply_file.relative_to(project_root)}: {size_kb:.1f} KB")
    print()


if __name__ == "__main__":
    main()
