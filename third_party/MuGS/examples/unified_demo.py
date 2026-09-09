#!/usr/bin/env python3
"""
MuGS Unified Demo

Simplified hybrid rendering demo using consolidated utilities.
Supports both multi-object and room scenes.

Usage:
    python unified_demo.py --scene kitchen_objects
    python unified_demo.py --scene kitchen_room --camera overview

Author: MuGS Team
Date: 2026-05-02
"""

import argparse
from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from mugs.utils.rendering import (
    load_ply_gaussians,
    render_mujoco_rgb,
    render_mujoco_segmentation,
    create_robot_mask,
    extract_mujoco_camera_params,
    composite_images
)


def simple_gaussian_rasterizer(
    gaussians_dict: dict,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_up: np.ndarray,
    fov: float,
    width: int,
    height: int
) -> np.ndarray:
    """CPU-based Gaussian rasterizer (fallback)."""
    # Create image
    image = np.ones((height, width, 3), dtype=np.float32) * 0.95

    # Camera coordinate system
    z_axis = camera_pos - camera_lookat
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(camera_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Camera matrix
    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = x_axis
    cam_to_world[:3, 1] = y_axis
    cam_to_world[:3, 2] = z_axis
    cam_to_world[:3, 3] = camera_pos

    world_to_cam = np.linalg.inv(cam_to_world)

    # Projection
    fov_rad = np.deg2rad(fov)
    f = height / (2 * np.tan(fov_rad / 2))

    # Render each object
    for obj_name, gaussians in gaussians_dict.items():
        means = gaussians['means']
        colors_sh = gaussians['sh_coeffs'][:, :3]  # DC component
        opacities = gaussians['opacities']

        # Convert SH to RGB (simplified)
        colors = (colors_sh + 0.5).clip(0, 1)

        # Transform to camera space
        means_hom = np.concatenate([means, np.ones((len(means), 1))], axis=1)
        means_cam = (world_to_cam @ means_hom.T).T[:, :3]

        # Project to screen
        x_screen = (means_cam[:, 0] / -means_cam[:, 2]) * f + width / 2
        y_screen = (means_cam[:, 1] / -means_cam[:, 2]) * f + height / 2
        depths = -means_cam[:, 2]

        # Filter visible points
        valid = (depths > 0) & (x_screen >= 0) & (x_screen < width) & \
                (y_screen >= 0) & (y_screen < height)

        x_screen = x_screen[valid]
        y_screen = y_screen[valid]
        depths = depths[valid]
        colors = colors[valid]
        opacities = opacities[valid]

        if len(x_screen) == 0:
            continue

        # Sort by depth (back to front)
        order = np.argsort(-depths)
        x_screen = x_screen[order]
        y_screen = y_screen[order]
        colors = colors[order]
        opacities = opacities[order]

        # Splat gaussians
        for i in range(len(x_screen)):
            cx, cy = int(x_screen[i]), int(y_screen[i])
            color = colors[i]
            alpha = opacities[i]

            # 5x5 splat kernel
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        weight = alpha * np.exp(-(dx*dx + dy*dy) / 2.0)
                        image[y, x] = image[y, x] * (1 - weight) + color * weight

    return image


def load_scene(scene_name: str, scene_dir: Path) -> dict:
    """Load 3DGS scene (multi-object or single room)."""
    gaussians_dict = {}

    if scene_name == "kitchen_objects":
        # Load multi-object scene
        object_files = {
            'coffee_mug': 'coffee_mug.ply',
            'dinner_plate': 'dinner_plate.ply',
            'cereal_bowl': 'cereal_bowl.ply',
            'water_glass': 'water_glass.ply',
        }

        for obj_name, filename in object_files.items():
            ply_path = scene_dir / filename
            if ply_path.exists():
                gaussians_dict[obj_name] = load_ply_gaussians(ply_path)

    elif scene_name == "kitchen_room":
        # Load single room PLY (if available)
        room_ply = scene_dir / "kitchen_real.ply"
        if room_ply.exists():
            gaussians_dict['kitchen_room'] = load_ply_gaussians(room_ply)
        else:
            print(f"⚠️  Room PLY not found: {room_ply}")
            print("   Using multi-object scene instead")
            return load_scene("kitchen_objects", scene_dir)

    return gaussians_dict


def main():
    parser = argparse.ArgumentParser(description="MuGS Unified Demo")
    parser.add_argument(
        '--scene',
        choices=['kitchen_objects', 'kitchen_room'],
        default='kitchen_objects',
        help='Scene type to render'
    )
    parser.add_argument(
        '--camera',
        default='overview',
        help='Camera name'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/unified_demo'),
        help='Output directory'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Image width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Image height'
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    scene_dir = project_root / "assets" / "scenes"
    xml_path = project_root / "assets" / "robot_arm_kitchen.xml"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MuGS Unified Hybrid Rendering Demo")
    print("="*70)
    print(f"Scene: {args.scene}")
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}×{args.height}")
    print()

    # Load scene
    print("📦 Loading 3DGS scene...")
    gaussians_dict = load_scene(args.scene, scene_dir)
    total_gaussians = sum(len(g['means']) for g in gaussians_dict.values())
    print(f"   Loaded {len(gaussians_dict)} objects, {total_gaussians:,} Gaussians")
    print()

    # Load MuJoCo
    print("🤖 Loading MuJoCo scene...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    print("   ✅ MuJoCo loaded")
    print()

    # Render pipeline
    print("🎨 Rendering hybrid scene...")

    # Step 1: MuJoCo RGB
    mujoco_rgb = render_mujoco_rgb(model, data, args.camera, args.width, args.height, renderer)
    print("   ✅ MuJoCo RGB rendered")

    # Step 2: Segmentation
    seg_ids = render_mujoco_segmentation(model, data, args.camera, args.width, args.height, renderer)
    print("   ✅ Segmentation rendered")

    # Step 3: Robot mask
    robot_geom_names = [
        "palm",
        "finger1_link", "finger2_link", "finger3_link",
        "finger4_link", "finger5_link"
    ]
    robot_mask = create_robot_mask(seg_ids, model, robot_geom_names)
    coverage = robot_mask.mean() * 100
    print(f"   ✅ Robot mask: {coverage:.1f}% coverage")

    # Step 4: 3DGS rendering
    camera_params = extract_mujoco_camera_params(model, data, args.camera, args.width, args.height)
    gs_rgb = simple_gaussian_rasterizer(
        gaussians_dict,
        camera_params['position'],
        camera_params['lookat'],
        camera_params['up'],
        camera_params['fov'],
        args.width,
        args.height
    )
    print("   ✅ 3DGS rendered (CPU)")

    # Step 5: Composite
    composite = composite_images(mujoco_rgb, gs_rgb, robot_mask)
    print("   ✅ Composite created")
    print()

    # Save outputs
    print("💾 Saving outputs...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(mujoco_rgb)
    axes[0, 0].set_title("MuJoCo RGB")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(seg_ids, cmap='tab20')
    axes[0, 1].set_title("Segmentation IDs")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(robot_mask, cmap='gray')
    axes[0, 2].set_title(f"Robot Mask ({coverage:.1f}%)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gs_rgb)
    axes[1, 0].set_title("3DGS Render")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(composite)
    axes[1, 1].set_title("Hybrid Composite")
    axes[1, 1].axis('off')

    # Difference map
    diff = np.abs(composite - gs_rgb).mean(axis=-1)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title("Difference (Composite - 3DGS)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = args.output_dir / f"{args.scene}_{args.camera}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved to {output_path}")
    print()
    print("="*70)
    print("✅ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
