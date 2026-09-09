#!/usr/bin/env python3
"""
Phase 1: Hybrid Rendering Test

Tests the hybrid rendering pipeline:
1. MuJoCo renders robot + segmentation mask
2. 3DGS renders objects/environment (placeholder)
3. Composite using mask
4. Save all intermediate outputs

Author: MuGS Team
Date: 2026-05-02
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "phase1_hybrid_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image resolution
WIDTH = 640
HEIGHT = 480

# Low-res for performance testing
WIDTH_LOW = 160
HEIGHT_LOW = 120


# ============================================================================
# MuJoCo Scene Setup
# ============================================================================


def create_test_scene_xml() -> str:
    """Create a simple test scene with robot hand and objects."""
    return """
    <mujoco model="hybrid_rendering_test">
        <option timestep="0.002" gravity="0 0 -9.81"/>

        <visual>
            <global offwidth="{width}" offheight="{height}"/>
            <quality shadowsize="4096"/>
        </visual>

        <asset>
            <!-- Materials for visualization -->
            <material name="table_mat" rgba="0.8 0.8 0.8 1"/>
            <material name="robot_mat" rgba="0.2 0.4 0.8 1"/>
            <material name="object_mat" rgba="0.8 0.2 0.2 1"/>

            <!-- Textures -->
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
            <material name="grid_mat" texture="grid" texrepeat="1 1"/>
        </asset>

        <worldbody>
            <!-- Lighting -->
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

            <!-- Ground plane -->
            <geom name="ground" type="plane" size="2 2 0.01"
                  material="grid_mat" rgba="0.9 0.9 0.9 1"/>

            <!-- Table -->
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

            <!-- Simple robot hand (5-finger gripper) -->
            <body name="robot_base" pos="0 -0.3 0.8">
                <!-- Palm -->
                <geom name="palm" type="box" size="0.04 0.06 0.02"
                      material="robot_mat" rgba="0.3 0.3 0.7 1"/>

                <!-- Finger 1 (thumb) -->
                <body name="finger1" pos="0.04 -0.04 0">
                    <joint name="finger1_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger1_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0.02 0 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 2 -->
                <body name="finger2" pos="0.02 0.04 0">
                    <joint name="finger2_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger2_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 3 -->
                <body name="finger3" pos="-0.02 0.04 0">
                    <joint name="finger3_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger3_link" type="capsule" size="0.008 0.03"
                          fromto="0 0 0 0 0.02 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 4 -->
                <body name="finger4" pos="0.01 0.06 0">
                    <joint name="finger4_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger4_link" type="capsule" size="0.008 0.025"
                          fromto="0 0 0 0 0.015 0" rgba="0.2 0.2 0.6 1"/>
                </body>

                <!-- Finger 5 -->
                <body name="finger5" pos="-0.01 0.06 0">
                    <joint name="finger5_joint" type="hinge" axis="0 0 1"
                           range="-0.5 0.5" damping="0.1"/>
                    <geom name="finger5_link" type="capsule" size="0.008 0.025"
                          fromto="0 0 0 0 0.015 0" rgba="0.2 0.2 0.6 1"/>
                </body>
            </body>

            <!-- Objects on table (these will be rendered by 3DGS in final version) -->
            <body name="mug" pos="0.1 0.0 0.46">
                <freejoint/>
                <geom name="mug_body" type="cylinder" size="0.035 0.06"
                      material="object_mat" rgba="0.3 0.6 0.8 1"/>
                <geom name="mug_handle" type="capsule" size="0.01 0.025"
                      fromto="0.04 0 -0.02 0.04 0 0.02"
                      rgba="0.3 0.6 0.8 1"/>
            </body>

            <body name="plate" pos="-0.15 0.0 0.43">
                <freejoint/>
                <geom name="plate_geom" type="cylinder" size="0.08 0.01"
                      rgba="1.0 1.0 1.0 1"/>
            </body>
        </worldbody>

        <!-- Camera -->
        <worldbody>
            <camera name="main_cam" pos="0.5 -0.5 1.0" xyaxes="1 0 0 0 1 1" mode="fixed"/>
            <camera name="front_cam" pos="0.0 -0.8 0.6" xyaxes="1 0 0 0 0.6 1" mode="fixed"/>
        </worldbody>
    </mujoco>
    """.format(width=WIDTH, height=HEIGHT)


# ============================================================================
# Rendering Functions
# ============================================================================


def render_mujoco_rgb(model, data, camera_name: str, width: int, height: int) -> np.ndarray:
    """Render RGB image from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)

    # Render RGB
    rgb = renderer.render()

    renderer.close()
    return rgb


def render_mujoco_segmentation(
    model, data, camera_name: str, width: int, height: int
) -> np.ndarray:
    """Render segmentation mask from MuJoCo."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera_name)

    # Enable segmentation rendering
    renderer.enable_segmentation_rendering()
    seg = renderer.render()

    # Convert to single channel (geom ID)
    # MuJoCo segmentation: [geom_id, ...]
    # We extract the geom IDs
    seg_ids = seg[:, :, 0].astype(np.int32)

    renderer.close()
    return seg_ids


def create_robot_mask(seg_ids: np.ndarray, model) -> np.ndarray:
    """
    Create binary mask for robot parts.

    Robot geoms: palm, finger1_link, finger2_link, finger3_link, finger4_link, finger5_link
    """
    robot_geom_names = [
        "palm",
        "finger1_link",
        "finger2_link",
        "finger3_link",
        "finger4_link",
        "finger5_link",
    ]

    # Get robot geom IDs
    robot_geom_ids = []
    for name in robot_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            robot_geom_ids.append(geom_id)

    # Create binary mask (1 = robot, 0 = environment/objects)
    mask = np.zeros_like(seg_ids, dtype=np.uint8)
    for geom_id in robot_geom_ids:
        mask[seg_ids == geom_id] = 1

    return mask


def render_3dgs_placeholder(width: int, height: int, object_poses: dict) -> np.ndarray:
    """
    Placeholder for 3DGS rendering.

    In Phase 1, we simulate 3DGS rendering with a simple procedural background.
    In Phase 2+, this will be replaced with actual gsplat rendering.

    Args:
        width: Image width
        height: Image height
        object_poses: Dict of object names to poses (unused in placeholder)

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    # Create a gradient background simulating environment lighting
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top = blue, bottom = white)
    for y in range(height):
        ratio = y / height
        sky_color = np.array([135, 206, 235]) * (1 - ratio) + np.array([255, 255, 255]) * ratio
        image[y, :] = sky_color

    # Add some "photorealistic" features (simulating 3DGS quality)
    # In real implementation, this will be gsplat.rasterize(...)

    # Placeholder: Draw colored circles for objects
    # Mug at (0.1, 0.0, 0.46) -> project to image space
    # For now, just draw at approximate pixel location
    mug_x, mug_y = int(width * 0.55), int(height * 0.5)
    cv2_available = False
    try:
        import cv2
        cv2_available = True
    except ImportError:
        pass

    if cv2_available:
        import cv2
        # Mug (blue)
        cv2.circle(image, (mug_x, mug_y), 40, (76, 153, 204), -1)
        cv2.circle(image, (mug_x, mug_y), 40, (50, 100, 150), 2)

        # Plate (white)
        plate_x, plate_y = int(width * 0.35), int(height * 0.55)
        cv2.ellipse(image, (plate_x, plate_y), (60, 50), 0, 0, 360, (240, 240, 240), -1)
        cv2.ellipse(image, (plate_x, plate_y), (60, 50), 0, 0, 360, (200, 200, 200), 2)

    return image


def composite_images(
    mujoco_rgb: np.ndarray,
    gs_rgb: np.ndarray,
    robot_mask: np.ndarray,
) -> np.ndarray:
    """
    Composite MuJoCo robot with 3DGS environment using mask.

    Args:
        mujoco_rgb: MuJoCo rendered image (H, W, 3)
        gs_rgb: 3DGS rendered image (H, W, 3)
        robot_mask: Binary mask (H, W), 1=robot, 0=environment

    Returns:
        Composited image (H, W, 3)
    """
    # Expand mask to 3 channels
    mask_3ch = np.stack([robot_mask] * 3, axis=-1)

    # Composite: where mask=1 use MuJoCo, where mask=0 use 3DGS
    composite = np.where(mask_3ch, mujoco_rgb, gs_rgb)

    return composite


# ============================================================================
# Visualization & Saving
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
    plt.imsave(output_dir / f"{prefix}4_3dgs_placeholder.png", gs_rgb)
    plt.imsave(output_dir / f"{prefix}5_composite.png", composite)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hybrid Rendering Pipeline", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(mujoco_rgb)
    axes[0, 0].set_title("1. MuJoCo RGB\n(Robot + Scene)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(seg_ids, cmap="tab20")
    axes[0, 1].set_title("2. Segmentation IDs\n(Geom IDs)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(robot_mask, cmap="gray")
    axes[0, 2].set_title("3. Robot Mask\n(Binary: 1=robot, 0=env)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gs_rgb)
    axes[1, 0].set_title("4. 3DGS Render\n(Environment + Objects)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(composite)
    axes[1, 1].set_title("5. Composite\n(Final Result)")
    axes[1, 1].axis("off")

    # Difference visualization
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
    print("Phase 1: Hybrid Rendering Test")
    print("="*60)
    print()

    # Create scene
    print("📝 Creating test scene...")
    xml_string = create_test_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Initialize simulation
    mujoco.mj_forward(model, data)

    print(f"   Scene: {model.nbody} bodies, {model.ngeom} geoms")
    print(f"   Cameras: {model.ncam}")

    # Test with both cameras
    cameras = ["main_cam", "front_cam"]

    for cam_name in cameras:
        print(f"\n📷 Rendering from camera: {cam_name}")
        print("-"*60)

        # Step 1: MuJoCo RGB
        print("   [1/5] Rendering MuJoCo RGB...")
        mujoco_rgb = render_mujoco_rgb(model, data, cam_name, WIDTH, HEIGHT)
        print(f"        Shape: {mujoco_rgb.shape}, dtype: {mujoco_rgb.dtype}")

        # Step 2: Segmentation
        print("   [2/5] Rendering segmentation mask...")
        seg_ids = render_mujoco_segmentation(model, data, cam_name, WIDTH, HEIGHT)
        print(f"        Shape: {seg_ids.shape}, unique IDs: {len(np.unique(seg_ids))}")

        # Step 3: Robot mask
        print("   [3/5] Creating robot mask...")
        robot_mask = create_robot_mask(seg_ids, model)
        robot_pixels = robot_mask.sum()
        total_pixels = robot_mask.size
        print(f"        Robot pixels: {robot_pixels}/{total_pixels} "
              f"({100*robot_pixels/total_pixels:.1f}%)")

        # Step 4: 3DGS rendering (placeholder)
        print("   [4/5] Rendering 3DGS (placeholder)...")
        object_poses = {}  # TODO: Extract from MuJoCo data
        gs_rgb = render_3dgs_placeholder(WIDTH, HEIGHT, object_poses)
        print(f"        Shape: {gs_rgb.shape}, dtype: {gs_rgb.dtype}")

        # Step 5: Composite
        print("   [5/5] Compositing images...")
        composite = composite_images(mujoco_rgb, gs_rgb, robot_mask)
        print(f"        Shape: {composite.shape}, dtype: {composite.dtype}")

        # Save outputs
        print("\n💾 Saving outputs...")
        save_outputs(
            mujoco_rgb, seg_ids, robot_mask, gs_rgb, composite,
            OUTPUT_DIR, prefix=cam_name
        )

    # Test low-resolution rendering for performance
    print(f"\n📊 Testing low-resolution rendering ({WIDTH_LOW}x{HEIGHT_LOW})...")
    print("-"*60)

    import time

    cam_name = "main_cam"
    n_iterations = 10

    times = []
    for i in range(n_iterations):
        start = time.time()

        mujoco_rgb_low = render_mujoco_rgb(model, data, cam_name, WIDTH_LOW, HEIGHT_LOW)
        seg_ids_low = render_mujoco_segmentation(model, data, cam_name, WIDTH_LOW, HEIGHT_LOW)
        robot_mask_low = create_robot_mask(seg_ids_low, model)
        gs_rgb_low = render_3dgs_placeholder(WIDTH_LOW, HEIGHT_LOW, {})
        composite_low = composite_images(mujoco_rgb_low, gs_rgb_low, robot_mask_low)

        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    print(f"   Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Throughput: {fps:.1f} FPS")
    print(f"   Target: 5000-8000 FPS (3DGS stage)")
    print(f"   Note: MuJoCo overhead currently dominates, 3DGS will be much faster")

    # Save low-res sample
    save_outputs(
        mujoco_rgb_low, seg_ids_low, robot_mask_low, gs_rgb_low, composite_low,
        OUTPUT_DIR, prefix="lowres"
    )

    print("\n" + "="*60)
    print("✅ Phase 1 Hybrid Rendering Test Complete!")
    print("="*60)
    print(f"\n📁 Outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")
    print("\n🚀 Next steps:")
    print("   1. Replace 3DGS placeholder with real gsplat rendering")
    print("   2. Implement super-resolution for low-res → high-res")
    print("   3. Benchmark full pipeline performance")
    print()


if __name__ == "__main__":
    main()
