#!/usr/bin/env python3
"""
Generate Complete Kitchen Scene with 3DGS Objects

Creates a realistic kitchen countertop scene with multiple objects.

Author: MuGS Team
Date: 2026-05-02
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


# ============================================================================
# Scene Configuration
# ============================================================================

KITCHEN_LAYOUT = {
    "counter": {
        "position": [0.0, 0.0, 0.0],  # Center at origin
        "size": [0.8, 0.6, 0.05],  # Width, depth, height (countertop)
    },

    "objects": [
        # Row 1: Back left
        {
            "name": "coffee_mug",
            "type": "cylinder",
            "position": [-0.25, 0.15, 0.08],
            "size": {"radius": 0.04, "height": 0.10},
            "color": [0.3, 0.2, 0.1],  # Brown
            "num_gaussians": 600,
        },
        {
            "name": "tea_cup",
            "type": "cylinder",
            "position": [-0.10, 0.18, 0.06],
            "size": {"radius": 0.03, "height": 0.08},
            "color": [0.9, 0.9, 0.9],  # White
            "num_gaussians": 500,
        },

        # Row 2: Back center
        {
            "name": "dinner_plate",
            "type": "cylinder",
            "position": [0.05, 0.12, 0.04],
            "size": {"radius": 0.12, "height": 0.02},
            "color": [0.95, 0.95, 0.95],  # White
            "num_gaussians": 800,
        },
        {
            "name": "side_plate",
            "type": "cylinder",
            "position": [0.25, 0.15, 0.04],
            "size": {"radius": 0.08, "height": 0.015},
            "color": [0.85, 0.9, 0.95],  # Light blue
            "num_gaussians": 600,
        },

        # Row 3: Front left
        {
            "name": "cereal_bowl",
            "type": "bowl",
            "position": [-0.20, -0.05, 0.06],
            "size": {"radius": 0.08, "depth": 0.05},
            "color": [0.4, 0.7, 0.3],  # Green
            "num_gaussians": 700,
        },
        {
            "name": "fruit_bowl",
            "type": "bowl",
            "position": [0.0, -0.08, 0.07],
            "size": {"radius": 0.10, "depth": 0.06},
            "color": [0.8, 0.6, 0.2],  # Orange/wood
            "num_gaussians": 800,
        },

        # Row 4: Front right
        {
            "name": "water_glass",
            "type": "cylinder",
            "position": [0.20, -0.10, 0.09],
            "size": {"radius": 0.03, "height": 0.12},
            "color": [0.7, 0.85, 0.9],  # Light blue (glass)
            "num_gaussians": 500,
        },
        {
            "name": "wine_glass",
            "type": "sphere",
            "position": [0.30, -0.05, 0.06],
            "size": {"radius": 0.04},
            "color": [0.75, 0.8, 0.85],  # Light blue (glass)
            "num_gaussians": 400,
        },

        # Center pieces
        {
            "name": "salt_shaker",
            "type": "cylinder",
            "position": [-0.05, 0.02, 0.07],
            "size": {"radius": 0.02, "height": 0.08},
            "color": [0.9, 0.9, 0.9],  # White
            "num_gaussians": 300,
        },
        {
            "name": "pepper_shaker",
            "type": "cylinder",
            "position": [0.05, 0.0, 0.07],
            "size": {"radius": 0.02, "height": 0.08},
            "color": [0.2, 0.2, 0.2],  # Black
            "num_gaussians": 300,
        },

        # Utensils
        {
            "name": "red_apple",
            "type": "sphere",
            "position": [-0.15, -0.15, 0.05],
            "size": {"radius": 0.04},
            "color": [0.8, 0.1, 0.1],  # Red
            "num_gaussians": 400,
        },
        {
            "name": "orange",
            "type": "sphere",
            "position": [0.15, -0.18, 0.04],
            "size": {"radius": 0.035},
            "color": [0.9, 0.5, 0.1],  # Orange
            "num_gaussians": 350,
        },
    ]
}


# ============================================================================
# Gaussian Generators
# ============================================================================


def create_sphere_gaussians(center, radius, num_gaussians, color):
    """Create Gaussians arranged in a sphere."""
    # Fibonacci sphere
    indices = np.arange(num_gaussians)
    phi = np.arccos(1 - 2 * (indices + 0.5) / num_gaussians)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    means = np.stack([x, y, z], axis=-1).astype(np.float32)
    scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.008
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.ones((num_gaussians, 1), dtype=np.float32) * 0.9

    sh_dc = np.zeros((num_gaussians, 3), dtype=np.float32)
    sh_dc[:] = (np.array(color) - 0.5) / 0.28209479177387814
    sh_rest = np.zeros((num_gaussians, 45), dtype=np.float32)

    return {
        'means': means, 'scales': scales, 'quats': quats,
        'opacities': opacities, 'sh_dc': sh_dc, 'sh_rest': sh_rest,
    }


def create_cylinder_gaussians(center, radius, height, num_gaussians, color):
    """Create Gaussians arranged in a cylinder."""
    num_height = int(np.sqrt(num_gaussians))
    num_circle = num_gaussians // num_height

    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    z = np.linspace(-height/2, height/2, num_height)

    theta_grid, z_grid = np.meshgrid(theta, z)
    theta_flat = theta_grid.flatten()
    z_flat = z_grid.flatten()

    actual_num = min(len(theta_flat), num_gaussians)
    theta_flat = theta_flat[:actual_num]
    z_flat = z_flat[:actual_num]

    x = radius * np.cos(theta_flat) + center[0]
    y = radius * np.sin(theta_flat) + center[1]
    z = z_flat + center[2]

    means = np.stack([x, y, z], axis=-1).astype(np.float32)
    scales = np.ones((actual_num, 3), dtype=np.float32) * 0.007
    quats = np.zeros((actual_num, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.ones((actual_num, 1), dtype=np.float32) * 0.95

    sh_dc = np.zeros((actual_num, 3), dtype=np.float32)
    sh_dc[:] = (np.array(color) - 0.5) / 0.28209479177387814
    sh_rest = np.zeros((actual_num, 45), dtype=np.float32)

    return {
        'means': means, 'scales': scales, 'quats': quats,
        'opacities': opacities, 'sh_dc': sh_dc, 'sh_rest': sh_rest,
    }


def create_bowl_gaussians(center, radius, depth, num_gaussians, color):
    """Create Gaussians arranged in a bowl (hemisphere)."""
    # Sphere but only bottom half
    indices = np.arange(num_gaussians * 2)
    phi = np.arccos(1 - 2 * (indices + 0.5) / (num_gaussians * 2))
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2] - radius + depth

    # Keep only bottom hemisphere
    mask = z < center[2]
    x, y, z = x[mask], y[mask], z[mask]

    # Take first num_gaussians
    x, y, z = x[:num_gaussians], y[:num_gaussians], z[:num_gaussians]
    actual_num = len(x)

    means = np.stack([x, y, z], axis=-1).astype(np.float32)
    scales = np.ones((actual_num, 3), dtype=np.float32) * 0.009
    quats = np.zeros((actual_num, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.ones((actual_num, 1), dtype=np.float32) * 0.92

    sh_dc = np.zeros((actual_num, 3), dtype=np.float32)
    sh_dc[:] = (np.array(color) - 0.5) / 0.28209479177387814
    sh_rest = np.zeros((actual_num, 45), dtype=np.float32)

    return {
        'means': means, 'scales': scales, 'quats': quats,
        'opacities': opacities, 'sh_dc': sh_dc, 'sh_rest': sh_rest,
    }


# ============================================================================
# PLY Export
# ============================================================================


def save_gaussians_to_ply(gaussians: dict, output_path: Path):
    """Save Gaussian parameters to PLY file."""
    num_gaussians = len(gaussians['means'])

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))

    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    vertex_data = np.zeros(num_gaussians, dtype=dtype)

    vertex_data['x'] = gaussians['means'][:, 0]
    vertex_data['y'] = gaussians['means'][:, 1]
    vertex_data['z'] = gaussians['means'][:, 2]
    vertex_data['nx'] = vertex_data['ny'] = vertex_data['nz'] = 0

    vertex_data['f_dc_0'] = gaussians['sh_dc'][:, 0]
    vertex_data['f_dc_1'] = gaussians['sh_dc'][:, 1]
    vertex_data['f_dc_2'] = gaussians['sh_dc'][:, 2]

    for i in range(45):
        vertex_data[f'f_rest_{i}'] = gaussians['sh_rest'][:, i]

    vertex_data['opacity'] = gaussians['opacities'][:, 0]
    vertex_data['scale_0'] = gaussians['scales'][:, 0]
    vertex_data['scale_1'] = gaussians['scales'][:, 1]
    vertex_data['scale_2'] = gaussians['scales'][:, 2]

    vertex_data['rot_0'] = gaussians['quats'][:, 0]
    vertex_data['rot_1'] = gaussians['quats'][:, 1]
    vertex_data['rot_2'] = gaussians['quats'][:, 2]
    vertex_data['rot_3'] = gaussians['quats'][:, 3]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([vertex_element]).write(output_path)

    return num_gaussians


# ============================================================================
# Scene Generation
# ============================================================================


def generate_kitchen_scene(output_dir: Path):
    """Generate complete kitchen scene with all objects."""
    print("="*60)
    print("Generating Kitchen Scene")
    print("="*60)
    print()

    scene_info = {
        "name": "kitchen_countertop",
        "description": "Complete kitchen scene with 12 objects",
        "objects": [],
        "total_gaussians": 0,
    }

    for obj_config in KITCHEN_LAYOUT["objects"]:
        name = obj_config["name"]
        obj_type = obj_config["type"]
        position = np.array(obj_config["position"])
        color = obj_config["color"]
        num_gaussians = obj_config["num_gaussians"]

        print(f"📦 Generating {name} ({obj_type})...")

        # Generate gaussians based on type
        if obj_type == "sphere":
            radius = obj_config["size"]["radius"]
            gaussians = create_sphere_gaussians(
                position, radius, num_gaussians, color
            )
        elif obj_type == "cylinder":
            radius = obj_config["size"]["radius"]
            height = obj_config["size"]["height"]
            gaussians = create_cylinder_gaussians(
                position, radius, height, num_gaussians, color
            )
        elif obj_type == "bowl":
            radius = obj_config["size"]["radius"]
            depth = obj_config["size"]["depth"]
            gaussians = create_bowl_gaussians(
                position, radius, depth, num_gaussians, color
            )
        else:
            print(f"   ⚠️  Unknown type: {obj_type}")
            continue

        # Save to PLY
        ply_path = output_dir / f"{name}.ply"
        actual_num = save_gaussians_to_ply(gaussians, ply_path)

        size_kb = ply_path.stat().st_size / 1024
        print(f"   ✅ {actual_num} Gaussians → {ply_path.name} ({size_kb:.1f} KB)")

        scene_info["objects"].append({
            "name": name,
            "type": obj_type,
            "position": position.tolist(),
            "num_gaussians": actual_num,
            "file": ply_path.name,
        })
        scene_info["total_gaussians"] += actual_num

    # Save scene metadata
    import json
    metadata_path = output_dir / "kitchen_scene.json"
    with open(metadata_path, 'w') as f:
        json.dump(scene_info, f, indent=2)

    print("\n" + "="*60)
    print("✅ Kitchen Scene Generated!")
    print("="*60)
    print(f"Total objects: {len(scene_info['objects'])}")
    print(f"Total Gaussians: {scene_info['total_gaussians']:,}")
    print(f"Scene metadata: {metadata_path}")
    print()

    return scene_info


# ============================================================================
# Main
# ============================================================================


def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "assets" / "scenes" / "kitchen"

    scene_info = generate_kitchen_scene(output_dir)

    print("Generated files:")
    for obj in scene_info["objects"]:
        print(f"   - {obj['file']}: {obj['num_gaussians']} Gaussians")
    print(f"   - kitchen_scene.json")
    print()


if __name__ == "__main__":
    main()
