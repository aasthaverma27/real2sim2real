#!/usr/bin/env python3
"""
Phase 1: Hybrid Rendering with Real 3DGS

Tests hybrid rendering with real gsplat rendering:
1. Load 3DGS PLY files (mug, plate)
2. MuJoCo renders robot + segmentation mask
3. gsplat renders objects/environment with real 3DGS models
4. Composite using mask
5. Save all intermediate outputs

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

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "phase1_real_gs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "objects"

# Image resolution
WIDTH = 640
HEIGHT = 480

# Low-res for performance testing
WIDTH_LOW = 160
HEIGHT_LOW = 120

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 3DGS PLY Loading
# ============================================================================


def load_ply_gaussians(ply_path: Path) -> dict:
    """
    Load 3DGS parameters from PLY file.

    Returns:
        Dict with tensors: means, scales, quats, opacities, sh_dc, sh_rest
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Extract positions
    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)

    # Extract scales
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    scales = np.exp(scales).astype(np.float32)  # Scales are stored as log

    # Extract rotations (quaternion: w, x, y, z -> x, y, z, w for gsplat)
    quats = np.stack([
        vertex['rot_1'],  # x
        vertex['rot_2'],  # y
        vertex['rot_3'],  # z
        vertex['rot_0'],  # w
    ], axis=-1).astype(np.float32)

    # Normalize quaternions
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    # Extract opacities
    opacities = vertex['opacity'].astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))  # Sigmoid activation
    opacities = opacities[:, None]  # (N, 1)

    # Extract SH coefficients
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    sh_dc = sh_dc.astype(np.float32)

    # Convert SH to RGB (simplified, just use DC component)
    # RGB = 0.5 + SH_C0 * sh_dc, where SH_C0 = 0.28209479177387814
    colors = 0.5 + 0.28209479177387814 * sh_dc
    colors = np.clip(colors, 0, 1)

    return {
        'means': torch.from_numpy(means).to(DEVICE),
        'scales': torch.from_numpy(scales).to(DEVICE),
        'quats': torch.from_numpy(quats).to(DEVICE),
        'opacities': torch.from_numpy(opacities).to(DEVICE),
        'colors': torch.from_numpy(colors).to(DEVICE),
    }


def transform_gaussians(gaussians: dict, position: np.ndarray, rotation: np.ndarray = None) -> dict:
    """
    Transform Gaussians to world coordinates.

    Args:
        gaussians: Dict with gaussian parameters
        position: (3,) translation
        rotation: (3, 3) rotation matrix (optional)

    Returns:
        Transformed gaussians dict
    """
    transformed = {}

    # Transform means
    means = gaussians['means'].cpu().numpy()
    if rotation is not None:
        means = means @ rotation.T
    means = means + position
    transformed['means'] = torch.from_numpy(means).to(DEVICE)

    # Transform quaternions if rotation provided
    if rotation is not None:
        # TODO: Proper quaternion multiplication
        # For now, keep original rotations
        transformed['quats'] = gaussians['quats']
    else:
        transformed['quats'] = gaussians['quats']

    # Copy other parameters
    transformed['scales'] = gaussians['scales']
    transformed['opacities'] = gaussians['opacities']
    transformed['colors'] = gaussians['colors']

    return transformed


# ============================================================================
# MuJoCo Scene (same as before)
# ============================================================================


def create_test_scene_xml() -> str:
    """Create a simple test scene."""
    return """
    <mujoco model="hybrid_rendering_real_gs">
        <option timestep="0.002" gravity="0 0 -9.81"/>

        <visual>
            <global offwidth="{width}" offheight="{height}"/>
            <quality shadowsize="4096"/>
        </visual>

        <asset>
            <material name="table_mat" rgba="0.8 0.8 0.8 1"/>
            <material name="robot_mat" rgba="0.2 0.4 0.8 1"/>

            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
            <material name="grid_mat" texture="grid" texrepeat="1 1"/>
        </asset>

        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

            <geom name="ground" type="plane" size="2 2 0.01"
                  material="grid_mat" rgba="0.9 0.9 0.9 1"/>

            <body name="table" pos="0 0 0.4">
                <geom name="table_top" type="box" size="0.4 0.3 0.02"
                      material="table_mat" rgba="0.6 0.4 0.2 1"/>
                <geom name="table_leg1" type="cylinder" size="0.02 0.2"
                      pos="0.35 0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg2" type="cylinder" size="0.02 0.2"
                      pos="0.35 -0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg3" type="cylinder" size="0.02 0.2"
                      pos="-0.35 0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
                <geom name="table_leg4" type="cylinder" size="0.02 0.2"
                      pos="-0.35 -0.25 -0.2" rgba="0.5 0.3 0.1 1"/>
            </body>

            <body name="robot_base" pos="0 -0.3 0.8">
                <geom name="palm" type="box" size="0.04 0.06 0.02"
                      material="robot_mat" rgba="0.3 0.3 0.7 1"/>

                <body name="finger1" pos="0.04 -0.04 0">
                    <joint name="finger1_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger1_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0.02 0 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger2" pos="0.02 0.04 0">
                    <joint name="finger2_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger2_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger3" pos="-0.02 0.04 0">
                    <joint name="finger3_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger3_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger4" pos="0.01 0.06 0">
                    <joint name="finger4_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger4_link" type="capsule" size="0.008 0.025"
                          fromto="0 0 0 0 0.015 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <body name="finger5" pos="-0.01 0.06 0">
                    <joint name="finger5_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger5_link" type="capsule" size="0.008 0.025"
                          fromto="0 0 0 0 0.015 0" rgba="0.2 0.2 0.6 1"/>
                </body>
            </body>

            <!-- Objects (invisible in MuJoCo, rendered by 3DGS) -->
            <body name="mug" pos="0.1 0.0 0.46">
                <freejoint/>
                <!-- Small invisible geom for physics only -->
                <geom name="mug_geom" type="cylinder" size="0.035 0.06"
                      rgba="0 0 0 0" group="1"/>
            </body>

            <body name="plate" pos="-0.15 0.0 0.43">
                <freejoint/>
                <geom name="plate_geom" type="cylinder" size="0.08 0.01"
                      rgba="0 0 0 0" group="1"/>
            </body>
        </worldbody>

        <worldbody>
            <camera name="main_cam" pos="0.5 -0.5 1.0" xyaxes="1 0 0 0 1 1" mode="fixed"/>
            <camera name="front_cam" pos="0.0 -0.8 0.6" xyaxes="1 0 0 0 0.6 1" mode="fixed"/>
        </worldbody>
    </mujoco>
    """.format(width=WIDTH, height=HEIGHT)


# ============================================================================
# MuJoCo Rendering (same as before)
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


# ============================================================================
# Real 3DGS Rendering with gsplat
# ============================================================================


def render_3dgs_real(
    gaussians_dict: dict,
    camera_pose: np.ndarray,
    camera_intrinsics: np.ndarray,
    width: int,
    height: int,
    background: torch.Tensor = None,
) -> np.ndarray:
    """
    Render scene using gsplat.

    Args:
        gaussians_dict: Dict of object_id -> gaussian parameters
        camera_pose: (4, 4) camera pose (world to camera)
        camera_intrinsics: (3, 3) camera intrinsics
        width: image width
        height: image height
        background: (3,) background color, default white

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    from gsplat import rasterization

    # Concatenate all gaussians
    all_means = []
    all_quats = []
    all_scales = []
    all_opacities = []
    all_colors = []

    for obj_id, gaussians in gaussians_dict.items():
        all_means.append(gaussians['means'])
        all_quats.append(gaussians['quats'])
        all_scales.append(gaussians['scales'])
        all_opacities.append(gaussians['opacities'])
        all_colors.append(gaussians['colors'])

    if not all_means:
        # No gaussians, return background
        if background is None:
            background = torch.ones(3, device=DEVICE)
        img = background.view(1, 1, 3).expand(height, width, 3)
        return (img.cpu().numpy() * 255).astype(np.uint8)

    means = torch.cat(all_means, dim=0)  # (N, 3)
    quats = torch.cat(all_quats, dim=0)  # (N, 4)
    scales = torch.cat(all_scales, dim=0)  # (N, 3)
    opacities = torch.cat(all_opacities, dim=0).squeeze(-1)  # (N,)
    colors = torch.cat(all_colors, dim=0)  # (N, 3)

    # Camera parameters
    viewmat = torch.from_numpy(camera_pose).float().to(DEVICE)
    K = torch.from_numpy(camera_intrinsics).float().to(DEVICE)

    # Background
    if background is None:
        background = torch.ones(3, device=DEVICE) * 0.95  # Light gray

    # Rasterize
    try:
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None, ...],  # (1, 4, 4)
            Ks=K[None, ...],  # (1, 3, 3)
            width=width,
            height=height,
            packed=False,
            backgrounds=background[None, ...],  # (1, 3)
        )

        # Extract image
        img = rendered[0]  # (H, W, 3)
        img = torch.clamp(img, 0, 1)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)

        return img_np

    except Exception as e:
        print(f"⚠️  gsplat rendering failed: {e}")
        print(f"   means: {means.shape}, quats: {quats.shape}")
        print(f"   scales: {scales.shape}, opacities: {opacities.shape}")
        # Return white image as fallback
        return np.ones((height, width, 3), dtype=np.uint8) * 240


def get_camera_matrices(model, data, camera_name: str, width: int, height: int):
    """Extract camera pose and intrinsics from MuJoCo."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # Camera position and orientation in world frame
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # MuJoCo camera looks along +Z, gsplat along -Z
    # Need to flip Z axis
    cam_mat[:, 2] *= -1

    # Create view matrix (world to camera)
    viewmat = np.eye(4)
    viewmat[:3, :3] = cam_mat.T
    viewmat[:3, 3] = -cam_mat.T @ cam_pos

    # Intrinsics (focal length and principal point)
    fovy = model.cam_fovy[cam_id]  # degrees
    fovy_rad = fovy * np.pi / 180
    fy = height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # Assume square pixels

    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return viewmat, K


# ============================================================================
# Composite (same as before)
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
# Visualization (updated with real 3DGS panel)
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
    plt.imsave(output_dir / f"{prefix}4_3dgs_real.png", gs_rgb)
    plt.imsave(output_dir / f"{prefix}5_composite.png", composite)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hybrid Rendering with Real 3DGS", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(mujoco_rgb)
    axes[0, 0].set_title("1. MuJoCo RGB\n(Robot + Scene)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_ids, cmap="tab20")
    axes[0, 1].set_title("2. Segmentation IDs")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(robot_mask, cmap="gray")
    axes[0, 2].set_title("3. Robot Mask\n(Binary)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gs_rgb)
    axes[1, 0].set_title("4. Real 3DGS Render\n(gsplat)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(composite)
    axes[1, 1].set_title("5. Composite\n(Final Result)")
    axes[1, 1].axis("off")

    # Difference
    diff = np.abs(mujoco_rgb.astype(float) - composite.astype(float)).mean(axis=-1)
    im = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("6. Difference\n(|MuJoCo - Composite|)")
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
    print("Phase 1: Hybrid Rendering with Real 3DGS")
    print("="*60)
    print(f"Device: {DEVICE}")
    print()

    # Load 3DGS assets
    print("📦 Loading 3DGS assets...")
    mug_ply = ASSETS_DIR / "kitchen" / "mug_blue.ply"
    plate_ply = ASSETS_DIR / "kitchen" / "plate_white.ply"

    if not mug_ply.exists():
        print(f"❌ Mug PLY not found: {mug_ply}")
        print("   Run: python scripts/utils/generate_simple_3dgs.py")
        return

    mug_gs = load_ply_gaussians(mug_ply)
    plate_gs = load_ply_gaussians(plate_ply)

    print(f"   Mug: {mug_gs['means'].shape[0]} Gaussians")
    print(f"   Plate: {plate_gs['means'].shape[0]} Gaussians")

    # Create MuJoCo scene
    print("\n📝 Creating MuJoCo scene...")
    xml_string = create_test_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"   Scene: {model.nbody} bodies, {model.ngeom} geoms")

    # Test with cameras
    cameras = ["main_cam", "front_cam"]

    for cam_name in cameras:
        print(f"\n📷 Rendering from camera: {cam_name}")
        print("-"*60)

        # Get object poses from MuJoCo
        mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

        mug_pos = data.xpos[mug_id].copy()
        plate_pos = data.xpos[plate_id].copy()

        print(f"   Mug position: {mug_pos}")
        print(f"   Plate position: {plate_pos}")

        # Transform gaussians to world coordinates
        mug_gs_world = transform_gaussians(mug_gs, mug_pos)
        plate_gs_world = transform_gaussians(plate_gs, plate_pos)

        gaussians_scene = {
            'mug': mug_gs_world,
            'plate': plate_gs_world,
        }

        # Get camera matrices
        viewmat, K = get_camera_matrices(model, data, cam_name, WIDTH, HEIGHT)

        # Step 1: MuJoCo RGB
        print("   [1/5] Rendering MuJoCo RGB...")
        mujoco_rgb = render_mujoco_rgb(model, data, cam_name, WIDTH, HEIGHT)

        # Step 2: Segmentation
        print("   [2/5] Rendering segmentation...")
        seg_ids = render_mujoco_segmentation(model, data, cam_name, WIDTH, HEIGHT)

        # Step 3: Robot mask
        print("   [3/5] Creating robot mask...")
        robot_mask = create_robot_mask(seg_ids, model)

        # Step 4: Real 3DGS rendering
        print("   [4/5] Rendering with gsplat...")
        gs_rgb = render_3dgs_real(gaussians_scene, viewmat, K, WIDTH, HEIGHT)

        # Step 5: Composite
        print("   [5/5] Compositing...")
        composite = composite_images(mujoco_rgb, gs_rgb, robot_mask)

        # Save
        print("\n💾 Saving outputs...")
        save_outputs(mujoco_rgb, seg_ids, robot_mask, gs_rgb, composite,
                    OUTPUT_DIR, prefix=cam_name)

    # Low-res test
    print(f"\n📊 Testing low-resolution ({WIDTH_LOW}x{HEIGHT_LOW})...")
    print("-"*60)

    import time

    cam_name = "main_cam"
    viewmat_low, K_low = get_camera_matrices(model, data, cam_name, WIDTH_LOW, HEIGHT_LOW)

    mug_gs_world = transform_gaussians(mug_gs, mug_pos)
    plate_gs_world = transform_gaussians(plate_gs, plate_pos)
    gaussians_scene = {'mug': mug_gs_world, 'plate': plate_gs_world}

    times = []
    n_iterations = 10

    for i in range(n_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        mujoco_rgb_low = render_mujoco_rgb(model, data, cam_name, WIDTH_LOW, HEIGHT_LOW)
        seg_ids_low = render_mujoco_segmentation(model, data, cam_name, WIDTH_LOW, HEIGHT_LOW)
        robot_mask_low = create_robot_mask(seg_ids_low, model)
        gs_rgb_low = render_3dgs_real(gaussians_scene, viewmat_low, K_low, WIDTH_LOW, HEIGHT_LOW)
        composite_low = composite_images(mujoco_rgb_low, gs_rgb_low, robot_mask_low)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    print(f"   Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Throughput: {fps:.1f} FPS")
    print(f"   Note: Still includes MuJoCo overhead")

    save_outputs(mujoco_rgb_low, seg_ids_low, robot_mask_low, gs_rgb_low, composite_low,
                OUTPUT_DIR, prefix="lowres")

    print("\n" + "="*60)
    print("✅ Real 3DGS Rendering Test Complete!")
    print("="*60)
    print(f"\n📁 Outputs: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")
    print()


if __name__ == "__main__":
    main()
