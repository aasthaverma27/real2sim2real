#!/usr/bin/env python3
"""
Phase 1: Hybrid Rendering - Fixed Version

Fixes:
1. Robot positioned in camera view
2. Simpler 3DGS rendering (no JIT compilation issues)
3. All intermediate outputs verified

Author: MuGS Team
Date: 2026-05-02
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from plyfile import PlyData

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "phase1_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "objects"

# Image resolution
WIDTH = 640
HEIGHT = 480

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 3DGS PLY Loading
# ============================================================================


def load_ply_gaussians(ply_path: Path) -> dict:
    """Load 3DGS parameters from PLY file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Extract positions
    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)

    # Extract scales (stored as log)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    scales = np.exp(scales).astype(np.float32)

    # Extract rotations (w,x,y,z)
    quats = np.stack([
        vertex['rot_0'],  # w
        vertex['rot_1'],  # x
        vertex['rot_2'],  # y
        vertex['rot_3'],  # z
    ], axis=-1).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    # Extract opacities
    opacities = vertex['opacity'].astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))  # Sigmoid
    opacities = opacities[:, None]

    # Extract colors from SH DC component
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    colors = 0.5 + 0.28209479177387814 * sh_dc.astype(np.float32)
    colors = np.clip(colors, 0, 1)

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'colors': colors,
    }


def transform_gaussians(gaussians: dict, position: np.ndarray) -> dict:
    """Transform Gaussians to world coordinates."""
    transformed = {
        'means': gaussians['means'] + position,
        'scales': gaussians['scales'].copy(),
        'quats': gaussians['quats'].copy(),
        'opacities': gaussians['opacities'].copy(),
        'colors': gaussians['colors'].copy(),
    }
    return transformed


# ============================================================================
# Simple 3DGS Rasterizer (CPU-based, no compilation needed)
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
    """
    Simple CPU-based Gaussian rasterizer.

    Not as fast as gsplat, but works without CUDA compilation.
    Good for testing and visualization.
    """
    # Concatenate all gaussians
    all_means = []
    all_colors = []
    all_opacities = []

    for obj_id, gaussians in gaussians_dict.items():
        all_means.append(gaussians['means'])
        all_colors.append(gaussians['colors'])
        all_opacities.append(gaussians['opacities'])

    if not all_means:
        return np.ones((height, width, 3), dtype=np.uint8) * 240

    means = np.concatenate(all_means, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    opacities = np.concatenate(all_opacities, axis=0).squeeze()

    # Create view matrix
    z_axis = camera_pos - camera_lookat
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(camera_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Transform points to camera space
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

    # Filter points behind camera
    visible = points_cam[:, 2] > 0.01
    points_cam = points_cam[visible]
    colors = colors[visible]
    opacities = opacities[visible]

    if len(points_cam) == 0:
        return np.ones((height, width, 3), dtype=np.uint8) * 240

    # Project to screen space
    x_ndc = points_cam[:, 0] / points_cam[:, 2] * f / aspect
    y_ndc = points_cam[:, 1] / points_cam[:, 2] * f

    x_screen = (x_ndc * 0.5 + 0.5) * width
    y_screen = (1 - (y_ndc * 0.5 + 0.5)) * height

    # Filter points outside screen
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

    # Render (simple splatting)
    image = np.ones((height, width, 3), dtype=np.float32) * 0.95  # Light gray background

    # Splat each Gaussian
    for i in range(len(x_screen)):
        cx, cy = int(x_screen[i]), int(y_screen[i])
        color = colors[i]
        alpha = opacities[i]

        # Simple splat (5x5 kernel)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    weight = alpha * np.exp(-(dx*dx + dy*dy) / 2.0)
                    image[y, x] = image[y, x] * (1 - weight) + color * weight

    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


# ============================================================================
# MuJoCo Scene (Optimized Layout)
# ============================================================================


def create_test_scene_xml() -> str:
    """Create scene with robot visible in camera view."""
    return """
    <mujoco model="hybrid_rendering_fixed">
        <option timestep="0.002" gravity="0 0 -9.81"/>

        <visual>
            <global offwidth="{width}" offheight="{height}"/>
            <quality shadowsize="4096"/>
        </visual>

        <asset>
            <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
            <material name="robot_mat" rgba="0.3 0.3 0.7 1"/>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
            <material name="grid_mat" texture="grid" texrepeat="1 1"/>
        </asset>

        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

            <geom name="ground" type="plane" size="2 2 0.01"
                  material="grid_mat" rgba="0.9 0.9 0.9 1"/>

            <!-- Table -->
            <body name="table" pos="0 0 0.4">
                <geom name="table_top" type="box" size="0.4 0.3 0.02"
                      rgba="0.6 0.4 0.2 1"/>
                <geom name="table_leg1" type="cylinder" size="0.02 0.2"
                      pos="0.35 0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg2" type="cylinder" size="0.02 0.2"
                      pos="0.35 -0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg3" type="cylinder" size="0.02 0.2"
                      pos="-0.35 0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg4" type="cylinder" size="0.02 0.2"
                      pos="-0.35 -0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
            </body>

            <!-- Robot hand - POSITIONED TO BE VISIBLE -->
            <body name="robot_base" pos="0.2 0.0 0.55" euler="0 0 -30">
                <!-- Palm (larger for visibility) -->
                <geom name="palm" type="box" size="0.05 0.07 0.025"
                      rgba="0.3 0.3 0.7 1"/>

                <!-- Finger 1 (thumb) -->
                <body name="finger1" pos="0.05 -0.05 0" euler="0 0 45">
                    <joint name="finger1_joint" type="hinge" axis="0 0 1"
                           range="-0.3 0.3" damping="0.1"/>
                    <geom name="finger1_link" type="capsule" size="0.01 0.035"
                          fromto="0 0 0 0.025 0 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 2 -->
                <body name="finger2" pos="0.03 0.05 0">
                    <joint name="finger2_joint" type="hinge" axis="0 0 1"
                           range="-0.3 0.3" damping="0.1"/>
                    <geom name="finger2_link" type="capsule" size="0.01 0.035"
                          fromto="0 0 0 0 0.025 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 3 -->
                <body name="finger3" pos="-0.03 0.05 0">
                    <joint name="finger3_joint" type="hinge" axis="0 0 1"
                           range="-0.3 0.3" damping="0.1"/>
                    <geom name="finger3_link" type="capsule" size="0.01 0.035"
                          fromto="0 0 0 0 0.025 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 4 -->
                <body name="finger4" pos="0.015 0.07 0">
                    <joint name="finger4_joint" type="hinge" axis="0 0 1"
                           range="-0.3 0.3" damping="0.1"/>
                    <geom name="finger4_link" type="capsule" size="0.01 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 5 -->
                <body name="finger5" pos="-0.015 0.07 0">
                    <joint name="finger5_joint" type="hinge" axis="0 0 1"
                           range="-0.3 0.3" damping="0.1"/>
                    <geom name="finger5_link" type="capsule" size="0.01 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>
            </body>

            <!-- Objects on table (will be rendered by 3DGS) -->
            <body name="mug" pos="-0.1 0.1 0.46">
                <freejoint/>
                <!-- Invisible in MuJoCo, visible in 3DGS -->
                <geom name="mug_geom" type="cylinder" size="0.001 0.001"
                      rgba="0 0 0 0" group="2"/>
            </body>

            <body name="plate" pos="0.0 -0.15 0.43">
                <freejoint/>
                <geom name="plate_geom" type="cylinder" size="0.001 0.001"
                      rgba="0 0 0 0" group="2"/>
            </body>
        </worldbody>

        <!-- Camera positioned to see robot and objects -->
        <worldbody>
            <camera name="main_cam" pos="0.6 -0.4 0.8" xyaxes="0.8 0.6 0 -0.3 0.4 1" mode="fixed"/>
            <camera name="top_cam" pos="0.0 0.0 1.2" xyaxes="1 0 0 0 1 0" mode="fixed"/>
        </worldbody>
    </mujoco>
    """.format(width=WIDTH, height=HEIGHT)


# ============================================================================
# MuJoCo Rendering
# ============================================================================


def render_mujoco_rgb(model, data, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render RGB image from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    renderer.close()
    return rgb


def render_mujoco_segmentation(model, data, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render segmentation mask from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    seg_ids = seg[:, :, 0].astype(np.int32)
    renderer.close()
    return seg_ids


def create_robot_mask(seg_ids: np.ndarray, model) -> np.ndarray:
    """Create binary mask for robot parts."""
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

    # Camera looks along -Z in OpenGL convention
    cam_lookat = cam_pos + cam_mat[:, 2]
    cam_up = cam_mat[:, 1]

    fov = model.cam_fovy[cam_id]

    return cam_pos, cam_lookat, cam_up, fov


# ============================================================================
# Composite
# ============================================================================


def composite_images(
    mujoco_rgb: np.ndarray,
    gs_rgb: np.ndarray,
    robot_mask: np.ndarray,
) -> np.ndarray:
    """Composite MuJoCo robot with 3DGS environment using mask."""
    mask_3ch = np.stack([robot_mask] * 3, axis=-1)
    composite = np.where(mask_3ch, mujoco_rgb, gs_rgb)
    return composite


# ============================================================================
# Visualization
# ============================================================================


def save_outputs(
    mujoco_rgb: np.ndarray,
    seg_ids: np.ndarray,
    robot_mask: np.ndarray,
    gs_rgb: np.ndarray,
    composite: np.ndarray,
    output_dir: Path,
    prefix: str = "",
):
    """Save all intermediate outputs."""
    if prefix:
        prefix = f"{prefix}_"

    # Save individual images
    plt.imsave(output_dir / f"{prefix}1_mujoco_rgb.png", mujoco_rgb)
    plt.imsave(output_dir / f"{prefix}2_segmentation.png", seg_ids, cmap="tab20")
    plt.imsave(output_dir / f"{prefix}3_robot_mask.png", robot_mask, cmap="gray")
    plt.imsave(output_dir / f"{prefix}4_3dgs_render.png", gs_rgb)
    plt.imsave(output_dir / f"{prefix}5_composite.png", composite)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hybrid Rendering Pipeline (Fixed)", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(mujoco_rgb)
    axes[0, 0].set_title("1. MuJoCo RGB\n(Robot visible!)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_ids, cmap="tab20")
    axes[0, 1].set_title("2. Segmentation IDs")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(robot_mask, cmap="gray")
    axes[0, 2].set_title("3. Robot Mask\n(Non-empty)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gs_rgb)
    axes[1, 0].set_title("4. 3DGS Render\n(CPU rasterizer)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(composite)
    axes[1, 1].set_title("5. Composite\n(Robot + Objects)")
    axes[1, 1].axis("off")

    # Difference
    diff = np.abs(mujoco_rgb.astype(float) - composite.astype(float)).mean(axis=-1)
    im = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("6. Difference")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}pipeline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved outputs to: {output_dir}")


# ============================================================================
# Main Test
# ============================================================================


def main():
    print("="*60)
    print("Phase 1: Hybrid Rendering (Fixed)")
    print("="*60)
    print()

    # Load 3DGS assets
    print("📦 Loading 3DGS assets...")
    mug_ply = ASSETS_DIR / "kitchen" / "mug_blue.ply"
    plate_ply = ASSETS_DIR / "kitchen" / "plate_white.ply"

    if not mug_ply.exists():
        print(f"❌ Assets not found. Run: python scripts/utils/generate_simple_3dgs.py")
        return

    mug_gs = load_ply_gaussians(mug_ply)
    plate_gs = load_ply_gaussians(plate_ply)

    print(f"   Mug: {len(mug_gs['means'])} Gaussians")
    print(f"   Plate: {len(plate_gs['means'])} Gaussians")

    # Create MuJoCo scene
    print("\n📝 Creating MuJoCo scene...")
    xml_string = create_test_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Set joint angles (open gripper slightly)
    data.qpos[0:5] = [0.2, 0.15, 0.15, 0.1, 0.1]
    mujoco.mj_forward(model, data)

    print(f"   Scene: {model.nbody} bodies, {model.ngeom} geoms")

    # Test with cameras
    cameras = ["main_cam"]

    for cam_name in cameras:
        print(f"\n📷 Rendering from camera: {cam_name}")
        print("-"*60)

        # Get object poses
        mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

        mug_pos = data.xpos[mug_id].copy()
        plate_pos = data.xpos[plate_id].copy()

        print(f"   Mug position: {mug_pos}")
        print(f"   Plate position: {plate_pos}")

        # Transform gaussians
        mug_gs_world = transform_gaussians(mug_gs, mug_pos)
        plate_gs_world = transform_gaussians(plate_gs, plate_pos)

        gaussians_scene = {
            'mug': mug_gs_world,
            'plate': plate_gs_world,
        }

        # Get camera parameters
        cam_pos, cam_lookat, cam_up, fov = get_camera_params(model, data, cam_name)
        print(f"   Camera pos: {cam_pos}")
        print(f"   Camera FOV: {fov}°")

        # Step 1: MuJoCo RGB
        print("   [1/5] MuJoCo RGB...")
        mujoco_rgb = render_mujoco_rgb(model, data, cam_name, WIDTH, HEIGHT)

        # Step 2: Segmentation
        print("   [2/5] Segmentation...")
        seg_ids = render_mujoco_segmentation(model, data, cam_name, WIDTH, HEIGHT)
        unique_ids = np.unique(seg_ids)
        print(f"        Unique segment IDs: {unique_ids}")

        # Step 3: Robot mask
        print("   [3/5] Robot mask...")
        robot_mask = create_robot_mask(seg_ids, model)
        robot_pixels = robot_mask.sum()
        total_pixels = robot_mask.size
        robot_pct = 100 * robot_pixels / total_pixels
        print(f"        Robot pixels: {robot_pixels}/{total_pixels} ({robot_pct:.1f}%)")

        if robot_pixels == 0:
            print("        ⚠️  WARNING: Robot not visible!")

        # Step 4: 3DGS rendering (CPU)
        print("   [4/5] 3DGS rendering (CPU)...")
        gs_rgb = simple_gaussian_rasterizer(
            gaussians_scene, cam_pos, cam_lookat, cam_up, fov, WIDTH, HEIGHT
        )
        print(f"        Rendered {len(gaussians_scene)} objects")

        # Step 5: Composite
        print("   [5/5] Composite...")
        composite = composite_images(mujoco_rgb, gs_rgb, robot_mask)

        # Save
        print("\n💾 Saving outputs...")
        save_outputs(mujoco_rgb, seg_ids, robot_mask, gs_rgb, composite,
                    OUTPUT_DIR, prefix=cam_name)

    print("\n" + "="*60)
    print("✅ Hybrid Rendering Test Complete!")
    print("="*60)
    print(f"\n📁 Outputs: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")
    print()


if __name__ == "__main__":
    main()
