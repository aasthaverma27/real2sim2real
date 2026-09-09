#!/usr/bin/env python3
"""
Kitchen Scene Demo - Complete Hybrid Rendering

Demonstrates full hybrid rendering pipeline with:
- Complete kitchen scene (12 objects, 6180 Gaussians)
- Robot hand interacting with objects
- MuJoCo physics + 3DGS rendering

Author: MuGS Team
Date: 2026-05-02
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from plyfile import PlyData

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "kitchen_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENE_DIR = PROJECT_ROOT / "assets" / "scenes" / "demo_kitchen"

WIDTH = 800
HEIGHT = 600


# ============================================================================
# 3DGS Loading
# ============================================================================


def load_ply_gaussians(ply_path: Path) -> dict:
    """Load 3DGS parameters from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    scales = np.exp(scales).astype(np.float32)

    quats = np.stack([
        vertex['rot_0'], vertex['rot_1'],
        vertex['rot_2'], vertex['rot_3'],
    ], axis=-1).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    opacities = vertex['opacity'].astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities[:, None]

    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    colors = 0.5 + 0.28209479177387814 * sh_dc.astype(np.float32)
    colors = np.clip(colors, 0, 1)

    return {
        'means': means, 'scales': scales, 'quats': quats,
        'opacities': opacities, 'colors': colors,
    }


def load_kitchen_scene(scene_dir: Path) -> dict:
    """Load complete kitchen scene."""
    metadata_path = scene_dir / "kitchen_scene.json"
    with open(metadata_path) as f:
        scene_info = json.load(f)

    print(f"📦 Loading kitchen scene: {scene_info['name']}")
    print(f"   Description: {scene_info['description']}")
    print(f"   Total objects: {len(scene_info['objects'])}")
    print(f"   Total Gaussians: {scene_info['total_gaussians']:,}")
    print()

    gaussians_dict = {}
    for obj in scene_info['objects']:
        name = obj['name']
        ply_path = scene_dir / obj['file']

        if not ply_path.exists():
            print(f"   ⚠️  Missing: {obj['file']}")
            continue

        gaussians = load_ply_gaussians(ply_path)
        # Transform to world position from scene metadata
        position = np.array(obj['position'])
        gaussians['means'] = gaussians['means'] + position

        gaussians_dict[name] = gaussians
        print(f"   ✅ {name}: {len(gaussians['means'])} Gaussians at {position}")

    print()
    return gaussians_dict


# ============================================================================
# Simple 3DGS Rasterizer
# ============================================================================


def simple_gaussian_rasterizer(
    gaussians_dict: dict,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_up: np.ndarray,
    fov: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Simple CPU-based Gaussian rasterizer."""
    # Concatenate all gaussians
    all_means, all_colors, all_opacities = [], [], []

    for obj_id, gaussians in gaussians_dict.items():
        all_means.append(gaussians['means'])
        all_colors.append(gaussians['colors'])
        all_opacities.append(gaussians['opacities'])

    if not all_means:
        return np.ones((height, width, 3), dtype=np.uint8) * 240

    means = np.concatenate(all_means, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    opacities = np.concatenate(all_opacities, axis=0).squeeze()

    # View matrix
    z_axis = camera_pos - camera_lookat
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(camera_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Transform to camera space
    points_world = means - camera_pos
    points_cam = np.stack([
        np.dot(points_world, x_axis),
        np.dot(points_world, y_axis),
        np.dot(points_world, z_axis),
    ], axis=-1)

    # Perspective projection
    fov_rad = fov * np.pi / 180
    f = 1.0 / np.tan(fov_rad / 2)
    aspect = width / height

    # Filter visible points
    visible = points_cam[:, 2] > 0.01
    points_cam = points_cam[visible]
    colors = colors[visible]
    opacities = opacities[visible]

    if len(points_cam) == 0:
        return np.ones((height, width, 3), dtype=np.uint8) * 240

    # Project to screen
    x_ndc = points_cam[:, 0] / points_cam[:, 2] * f / aspect
    y_ndc = points_cam[:, 1] / points_cam[:, 2] * f

    x_screen = (x_ndc * 0.5 + 0.5) * width
    y_screen = (1 - (y_ndc * 0.5 + 0.5)) * height

    # Filter in-screen points
    in_screen = (
        (x_screen >= 0) & (x_screen < width) &
        (y_screen >= 0) & (y_screen < height)
    )
    x_screen = x_screen[in_screen]
    y_screen = y_screen[in_screen]
    colors = colors[in_screen]
    opacities = opacities[in_screen]
    depths = points_cam[in_screen, 2]

    # Sort by depth (back to front)
    depth_order = np.argsort(-depths)
    x_screen = x_screen[depth_order]
    y_screen = y_screen[depth_order]
    colors = colors[depth_order]
    opacities = opacities[depth_order]

    # Render
    image = np.ones((height, width, 3), dtype=np.float32) * 0.95

    for i in range(len(x_screen)):
        cx, cy = int(x_screen[i]), int(y_screen[i])
        color = colors[i]
        alpha = opacities[i]

        # 5x5 splat
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    weight = alpha * np.exp(-(dx*dx + dy*dy) / 2.0)
                    image[y, x] = image[y, x] * (1 - weight) + color * weight

    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


# ============================================================================
# MuJoCo Scene
# ============================================================================


def create_kitchen_mujoco_scene() -> str:
    """Create MuJoCo scene with robot and kitchen countertop."""
    return """
    <mujoco model="kitchen_demo">
        <option timestep="0.002" gravity="0 0 -9.81"/>

        <visual>
            <global offwidth="{width}" offheight="{height}"/>
            <quality shadowsize="4096"/>
        </visual>

        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
            <material name="grid_mat" texture="grid" texrepeat="3 3"/>
            <material name="counter_mat" rgba="0.7 0.5 0.3 1"/>
        </asset>

        <worldbody>
            <light pos="0 0 2" dir="0 0 -1" diffuse="1.0 1.0 1.0"/>
            <light pos="1 1 1.5" dir="-1 -1 -1" diffuse="0.6 0.6 0.6"/>

            <!-- Ground -->
            <geom name="ground" type="plane" size="3 3 0.01"
                  material="grid_mat" rgba="0.9 0.9 0.9 1"/>

            <!-- Kitchen counter (matches scene generator) -->
            <body name="counter" pos="0 0 0.4">
                <geom name="counter_top" type="box" size="0.8 0.6 0.05"
                      rgba="0.7 0.5 0.3 1"/>
                <!-- Legs -->
                <geom name="leg1" type="cylinder" size="0.03 0.2"
                      pos="0.7 0.5 -0.25" rgba="0.6 0.4 0.2 1"/>
                <geom name="leg2" type="cylinder" size="0.03 0.2"
                      pos="0.7 -0.5 -0.25" rgba="0.6 0.4 0.2 1"/>
                <geom name="leg3" type="cylinder" size="0.03 0.2"
                      pos="-0.7 0.5 -0.25" rgba="0.6 0.4 0.2 1"/>
                <geom name="leg4" type="cylinder" size="0.03 0.2"
                      pos="-0.7 -0.5 -0.25" rgba="0.6 0.4 0.2 1"/>
            </body>

            <!-- Robot hand reaching over counter -->
            <body name="robot_base" pos="0.5 0.3 0.8" euler="0 20 -45">
                <geom name="palm" type="box" size="0.06 0.08 0.03"
                      rgba="0.3 0.3 0.7 1"/>

                <body name="finger1" pos="0.06 -0.06 0" euler="0 0 40">
                    <joint name="finger1_joint" type="hinge" axis="0 0 1"
                           range="-0.4 0.4" damping="0.1"/>
                    <geom name="finger1_link" type="capsule" size="0.012 0.04"
                          fromto="0 0 0 0.03 0 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger2" pos="0.04 0.06 0" euler="0 0 -10">
                    <joint name="finger2_joint" type="hinge" axis="0 0 1"
                           range="-0.4 0.4" damping="0.1"/>
                    <geom name="finger2_link" type="capsule" size="0.012 0.04"
                          fromto="0 0 0 0 0.03 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger3" pos="-0.04 0.06 0" euler="0 0 -10">
                    <joint name="finger3_joint" type="hinge" axis="0 0 1"
                           range="-0.4 0.4" damping="0.1"/>
                    <geom name="finger3_link" type="capsule" size="0.012 0.04"
                          fromto="0 0 0 0 0.03 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger4" pos="0.02 0.08 0" euler="0 0 -10">
                    <joint name="finger4_joint" type="hinge" axis="0 0 1"
                           range="-0.4 0.4" damping="0.1"/>
                    <geom name="finger4_link" type="capsule" size="0.012 0.035"
                          fromto="0 0 0 0 0.025 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger5" pos="-0.02 0.08 0" euler="0 0 -10">
                    <joint name="finger5_joint" type="hinge" axis="0 0 1"
                           range="-0.4 0.4" damping="0.1"/>
                    <geom name="finger5_link" type="capsule" size="0.012 0.035"
                          fromto="0 0 0 0 0.025 0" rgba="0.2 0.2 0.6 1"/>
                </body>
            </body>
        </worldbody>

        <!-- Cameras -->
        <worldbody>
            <camera name="overview" pos="1.2 -0.8 1.2" xyaxes="0.8 0.6 0 -0.3 0.4 1" mode="fixed"/>
            <camera name="closeup" pos="0.4 -0.3 0.9" xyaxes="0.9 0.4 0 -0.2 0.5 1" mode="fixed"/>
            <camera name="top_down" pos="0.0 0.0 1.5" xyaxes="1 0 0 0 1 0" mode="fixed"/>
        </worldbody>
    </mujoco>
    """.format(width=WIDTH, height=HEIGHT)


# ============================================================================
# MuJoCo Rendering
# ============================================================================


def render_mujoco_rgb(model, data, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render RGB from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    renderer.close()
    return rgb


def render_mujoco_segmentation(model, data, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render segmentation from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    seg_ids = seg[:, :, 0].astype(np.int32)
    renderer.close()
    return seg_ids


def create_robot_mask(seg_ids: np.ndarray, model) -> np.ndarray:
    """Create robot mask."""
    robot_geom_names = [
        "palm", "finger1_link", "finger2_link",
        "finger3_link", "finger4_link", "finger5_link",
    ]

    robot_geom_ids = []
    for name in robot_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            robot_geom_ids.append(geom_id)

    mask = np.zeros_like(seg_ids, dtype=np.uint8)
    for geom_id in robot_geom_ids:
        mask[seg_ids == geom_id] = 1

    return mask


def get_camera_params(model, data, camera_name: str):
    """Extract camera parameters."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_lookat = cam_pos + cam_mat[:, 2]
    cam_up = cam_mat[:, 1]
    fov = model.cam_fovy[cam_id]
    return cam_pos, cam_lookat, cam_up, fov


def composite_images(mujoco_rgb, gs_rgb, robot_mask):
    """Composite images."""
    mask_3ch = np.stack([robot_mask] * 3, axis=-1)
    return np.where(mask_3ch, mujoco_rgb, gs_rgb)


# ============================================================================
# Visualization
# ============================================================================


def save_demo_output(mujoco_rgb, seg_ids, robot_mask, gs_rgb, composite,
                     output_dir, prefix=""):
    """Save demo outputs."""
    if prefix:
        prefix = f"{prefix}_"

    plt.imsave(output_dir / f"{prefix}1_mujoco.png", mujoco_rgb)
    plt.imsave(output_dir / f"{prefix}2_segmentation.png", seg_ids, cmap="tab20")
    plt.imsave(output_dir / f"{prefix}3_robot_mask.png", robot_mask, cmap="gray")
    plt.imsave(output_dir / f"{prefix}4_3dgs_kitchen.png", gs_rgb)
    plt.imsave(output_dir / f"{prefix}5_composite.png", composite)

    # Create demo figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Kitchen Scene Demo - Hybrid Rendering", fontsize=18, fontweight="bold")

    axes[0, 0].imshow(mujoco_rgb)
    axes[0, 0].set_title("1. MuJoCo\n(Robot + Counter)", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_ids, cmap="tab20")
    axes[0, 1].set_title("2. Segmentation", fontsize=14)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(robot_mask, cmap="gray")
    axes[0, 2].set_title("3. Robot Mask", fontsize=14)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gs_rgb)
    axes[1, 0].set_title("4. 3DGS Kitchen Scene\n(12 objects, 6180 Gaussians)", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(composite)
    axes[1, 1].set_title("5. Final Composite\n(Robot + Kitchen)", fontsize=14)
    axes[1, 1].axis("off")

    diff = np.abs(mujoco_rgb.astype(float) - composite.astype(float)).mean(axis=-1)
    im = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("6. Difference Map", fontsize=14)
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}demo_complete.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main Demo
# ============================================================================


def main():
    print("="*70)
    print("Kitchen Scene Demo - Complete Hybrid Rendering")
    print("="*70)
    print()

    # Load kitchen scene (3DGS objects)
    print("STEP 1: Loading 3DGS Kitchen Scene")
    print("-"*70)
    gaussians_dict = load_kitchen_scene(SCENE_DIR)
    total_gaussians = sum(len(g['means']) for g in gaussians_dict.values())
    print(f"✅ Loaded {len(gaussians_dict)} objects with {total_gaussians:,} total Gaussians\n")

    # Create MuJoCo scene (robot + counter)
    print("STEP 2: Creating MuJoCo Scene")
    print("-"*70)
    xml_string = create_kitchen_mujoco_scene()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Set gripper pose (reaching towards cup)
    data.qpos[0:5] = [0.3, 0.2, 0.2, 0.15, 0.15]
    mujoco.mj_forward(model, data)
    print(f"✅ MuJoCo scene created ({model.nbody} bodies, {model.ngeom} geoms)\n")

    # Render from multiple cameras
    cameras = ["overview", "closeup", "top_down"]

    for cam_name in cameras:
        print(f"STEP 3: Rendering from camera '{cam_name}'")
        print("-"*70)

        # Get camera params
        cam_pos, cam_lookat, cam_up, fov = get_camera_params(model, data, cam_name)
        print(f"Camera position: {cam_pos}")
        print(f"FOV: {fov}°")

        # MuJoCo render
        print("   [1/5] MuJoCo RGB...")
        mujoco_rgb = render_mujoco_rgb(model, data, cam_name, WIDTH, HEIGHT)

        # Segmentation
        print("   [2/5] Segmentation...")
        seg_ids = render_mujoco_segmentation(model, data, cam_name, WIDTH, HEIGHT)

        # Robot mask
        print("   [3/5] Robot mask...")
        robot_mask = create_robot_mask(seg_ids, model)
        robot_pct = 100 * robot_mask.sum() / robot_mask.size
        print(f"        Robot coverage: {robot_pct:.1f}%")

        # 3DGS render
        print("   [4/5] 3DGS kitchen scene...")
        gs_rgb = simple_gaussian_rasterizer(
            gaussians_dict, cam_pos, cam_lookat, cam_up, fov, WIDTH, HEIGHT
        )

        # Composite
        print("   [5/5] Composite...")
        composite = composite_images(mujoco_rgb, gs_rgb, robot_mask)

        # Save
        print(f"   💾 Saving outputs...")
        save_demo_output(mujoco_rgb, seg_ids, robot_mask, gs_rgb, composite,
                        OUTPUT_DIR, prefix=cam_name)
        print()

    print("="*70)
    print("✅ Kitchen Scene Demo Complete!")
    print("="*70)
    print(f"\n📁 Outputs: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
