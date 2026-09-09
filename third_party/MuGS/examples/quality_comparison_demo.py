#!/usr/bin/env python3
"""
Quality Comparison: Native RGB vs MuGS Hybrid

Compare rendering quality between:
1. MuJoCo native RGB (baseline camera render)
2. MuGS Hybrid (3DGS background + MuJoCo foreground)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from mugs.sensors import GaussianSensor, GaussianSensorConfig


def create_pick_place_scene():
    """Create a realistic pick-place scene with YAM-like arm."""
    xml = """
    <mujoco model="pick_place_quality">
      <compiler angle="radian"/>
      <option timestep="0.005" gravity="0 0 -9.81"/>

      <visual>
        <global offwidth="640" offheight="480"/>
        <quality shadowsize="4096"/>
        <map stiffness="100" shadowscale="0.5"/>
        <headlight ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8"/>
      </visual>

      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
        <material name="grid" texture="grid" texrepeat="2 2"/>

        <texture name="metal" type="cube" builtin="flat" mark="cross" width="127" height="127"
                 rgb1="0.5 0.5 0.5" rgb2="0.6 0.6 0.6" markrgb="1 1 1"/>
        <material name="metal" texture="metal" specular="1" shininess="0.3"/>
      </asset>

      <worldbody>
        <!-- Lighting (much brighter for clear visibility) -->
        <light pos="0.4 0 1.5" dir="0 0 -1" directional="false" castshadow="true" diffuse="0.8 0.8 0.8"/>
        <light pos="0 -0.5 1.2" dir="0 0.5 -1" directional="false" diffuse="0.6 0.6 0.6"/>
        <light pos="0.8 0.2 1.0" dir="-0.5 0 -1" directional="false" diffuse="0.5 0.5 0.5"/>

        <!-- Floor -->
        <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>

        <!-- Table (invisible collision, will be 3DGS background) -->
        <geom name="table" type="box" size="0.4 0.3 0.02" pos="0.4 0 0.6" rgba="0.5 0.4 0.3 0"/>

        <!-- First-person camera (wrist-mounted, looking at objects on table) -->
        <camera name="wrist_view" pos="0.25 -0.15 0.75"
                xyaxes="0.9 0.4 0 -0.25 0.55 0.8" fovy="0.9599"/>

        <!-- Third-person camera (side view showing full scene) -->
        <camera name="side_view" pos="0.9 -0.6 0.8"
                xyaxes="0.8 0.6 0 -0.35 0.5 0.8" fovy="0.7854"/>

        <!-- Robot arm (4-DOF for realistic manipulation) -->
        <body name="robot_base" pos="0 0 0.6">
          <geom name="base" type="cylinder" size="0.06 0.03" rgba="0.3 0.3 0.35 1" material="metal"/>

          <!-- Shoulder -->
          <body name="shoulder" pos="0 0 0.05">
            <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom name="shoulder_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.12"
                  rgba="0.6 0.6 0.65 1" material="metal"/>

            <!-- Upper arm -->
            <body name="upper_arm" pos="0 0 0.12">
              <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
              <geom name="upper_arm_geom" type="capsule" size="0.03" fromto="0 0 0 0.18 0 0"
                    rgba="0.5 0.5 0.55 1" material="metal"/>

              <!-- Forearm -->
              <body name="forearm" pos="0.18 0 0">
                <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <geom name="forearm_geom" type="capsule" size="0.025" fromto="0 0 0 0.15 0 0"
                      rgba="0.4 0.4 0.45 1" material="metal"/>

                <!-- Wrist -->
                <body name="wrist" pos="0.15 0 0">
                  <joint name="wrist_rotate" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
                  <geom name="wrist_geom" type="cylinder" size="0.02 0.03"
                        rgba="0.35 0.35 0.4 1" material="metal"/>

                  <!-- Gripper -->
                  <body name="gripper_base" pos="0.04 0 0">
                    <geom name="palm" type="box" size="0.015 0.025 0.01"
                          rgba="0.3 0.3 0.35 1"/>

                    <!-- Left finger -->
                    <body name="left_finger" pos="-0.02 0 0">
                      <joint name="left_finger_joint" type="slide" axis="1 0 0"
                             range="-0.025 0" damping="0.1"/>
                      <geom name="left_finger" type="box" size="0.004 0.015 0.02"
                            rgba="0.4 0.4 0.45 1"/>
                    </body>

                    <!-- Right finger -->
                    <body name="right_finger" pos="0.02 0 0">
                      <joint name="right_finger_joint" type="slide" axis="1 0 0"
                             range="0 0.025" damping="0.1"/>
                      <geom name="right_finger" type="box" size="0.004 0.015 0.02"
                            rgba="0.4 0.4 0.45 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- Objects to pick -->
        <!-- Red cube -->
        <body name="red_cube" pos="0.35 0.05 0.63">
          <joint type="free"/>
          <geom name="red_cube_geom" type="box" size="0.02 0.02 0.02"
                rgba="0.9 0.2 0.15 1" mass="0.05"/>
        </body>

        <!-- Blue cylinder -->
        <body name="blue_cylinder" pos="0.45 -0.08 0.635">
          <joint type="free"/>
          <geom name="blue_cylinder_geom" type="cylinder" size="0.015 0.035"
                rgba="0.15 0.4 0.85 1" mass="0.04"/>
        </body>

        <!-- Green sphere -->
        <body name="green_sphere" pos="0.5 0.12 0.63">
          <joint type="free"/>
          <geom name="green_sphere_geom" type="sphere" size="0.018"
                rgba="0.2 0.75 0.3 1" mass="0.03"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="shoulder_pan_motor" joint="shoulder_pan" gear="20"/>
        <motor name="shoulder_lift_motor" joint="shoulder_lift" gear="20"/>
        <motor name="elbow_motor" joint="elbow" gear="15"/>
        <motor name="wrist_rotate_motor" joint="wrist_rotate" gear="10"/>
        <motor name="left_finger_motor" joint="left_finger_joint" gear="5"/>
        <motor name="right_finger_motor" joint="right_finger_joint" gear="5"/>
      </actuator>
    </mujoco>
    """
    return xml


def main():
    print("=" * 80)
    print("Quality Comparison: Native RGB vs MuGS Hybrid")
    print("=" * 80)

    # Create scene
    scene_xml = create_pick_place_scene()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(scene_xml)
        scene_path = f.name

    print(f"✅ Scene created: {scene_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Set robot to reaching pose
    data.qpos[0] = 0.3   # shoulder pan
    data.qpos[1] = 0.5   # shoulder lift
    data.qpos[2] = -0.8  # elbow
    data.qpos[3] = 0.2   # wrist rotate
    data.qpos[4] = -0.015  # left finger
    data.qpos[5] = 0.015   # right finger

    mujoco.mj_forward(model, data)

    # Setup paths
    ply_path = Path(__file__).parent.parent / "data" / "pretrained" / "kitchen" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    output_dir = Path(__file__).parent.parent / "outputs" / "quality_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        print(f"❌ Background not found: {ply_path}")
        return

    print(f"✅ Background: {ply_path}")

    # Robot geoms
    robot_geoms = [
        'base', 'shoulder_geom', 'upper_arm_geom', 'forearm_geom',
        'wrist_geom', 'palm', 'left_finger', 'right_finger',
        'red_cube_geom', 'blue_cylinder_geom', 'green_sphere_geom',
    ]

    print(f"✅ Foreground geoms: {len(robot_geoms)}")

    # Render with MuJoCo native (baseline)
    print("\n" + "=" * 80)
    print("Rendering with MuJoCo native renderer...")
    print("=" * 80)

    renderer_native = mujoco.Renderer(model, 480, 640)
    mujoco.mj_forward(model, data)
    renderer_native.update_scene(data, camera="wrist_view")
    native_rgb = renderer_native.render()
    renderer_native.close()

    print(f"✅ Native RGB shape: {native_rgb.shape}")

    # Render with MuGS
    print("\n" + "=" * 80)
    print("Rendering with MuGS hybrid...")
    print("=" * 80)

    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=ply_path,
        render_mode="hybrid",
        robot_geom_names=robot_geoms,
    )
    sensor = GaussianSensor(config)

    result = sensor.render(model, data, "wrist_view", return_components=True)

    print(f"✅ MuGS RGB shape: {result['rgb'].shape}")

    # Also render MuJoCo-only mode for reference
    sensor.cfg.render_mode = "mujoco_only"
    mujoco_only = sensor.render(model, data, "wrist_view")

    # Visualization
    print("\n" + "=" * 80)
    print("Generating quality comparison...")
    print("=" * 80)

    fig = plt.figure(figsize=(20, 12))

    # Row 1: Main comparison
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(native_rgb)
    ax1.set_title('MuJoCo Native RGB\n(Task Camera)', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(mujoco_only)
    ax2.set_title('MuJoCo Only\n(MuGS Pipeline)', fontsize=13, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(result['background'])
    ax3.set_title('3DGS Background\n(Photorealistic)', fontsize=13, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(result['rgb'])
    ax4.set_title('MuGS Hybrid\n(Final Result)', fontsize=13, fontweight='bold', color='#d62728')
    ax4.axis('off')

    # Row 2: Details
    ax5 = plt.subplot(2, 4, 5)
    # Zoom on robot (crop center region)
    h, w = native_rgb.shape[:2]
    crop = native_rgb[h//4:3*h//4, w//4:3*w//4]
    ax5.imshow(crop)
    ax5.set_title('Native - Robot Detail', fontsize=11)
    ax5.axis('off')

    ax6 = plt.subplot(2, 4, 6)
    crop_mugs = result['rgb'][h//4:3*h//4, w//4:3*w//4]
    ax6.imshow(crop_mugs)
    ax6.set_title('MuGS - Robot Detail', fontsize=11)
    ax6.axis('off')

    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(result['foreground'])
    ax7.set_title('MuGS Foreground\n(Robot + Objects)', fontsize=11)
    ax7.axis('off')

    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(result['mask'], cmap='hot')
    ax8.set_title('MuGS Mask\n(Blending Weights)', fontsize=11)
    ax8.axis('off')

    plt.suptitle('Quality Comparison: Pick-Place Task Rendering',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    output_path = output_dir / "quality_comparison.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Copy to outputs
    import shutil
    shutil.copy(output_path, "/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250/quality_comparison.jpg")
    print("✅ Copied to user outputs")

    # Save individual images for detailed inspection
    plt.imsave(output_dir / "native_rgb.jpg", native_rgb)
    plt.imsave(output_dir / "mugs_hybrid.jpg", result['rgb'])
    plt.imsave(output_dir / "mugs_background.jpg", result['background'])
    plt.imsave(output_dir / "mugs_foreground.jpg", result['foreground'])

    print("\n" + "=" * 80)
    print("Quality Analysis:")
    print("=" * 80)

    # Calculate quality metrics
    robot_ratio = result['mask'].mean()

    print(f"  Resolution: {result['rgb'].shape[1]}x{result['rgb'].shape[0]}")
    print(f"  Robot coverage: {robot_ratio*100:.2f}%")
    print(f"  Background coverage: {(1-robot_ratio)*100:.2f}%")
    print(f"\n  Native RGB: Standard MuJoCo render (grid floor, simple lighting)")
    print(f"  MuGS Hybrid: Same robot/objects + photorealistic kitchen background")
    print(f"\n  Quality gain: Photorealistic environment with accurate physics!")

    print("\n" + "=" * 80)
    print("✅ Quality comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
