#!/usr/bin/env python3
"""
Simple YAM + MuGS Demo (Standalone Mode)

Test GaussianSensor on YAM robot without full mjlab environment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from mugs.sensors import GaussianSensor, GaussianSensorConfig


def main():
    print("=" * 80)
    print("YAM + MuGS Standalone Demo")
    print("=" * 80)

    # Create simple YAM scene XML
    yam_scene_xml = """
    <mujoco model="yam_simple">
      <compiler angle="radian"/>
      <option timestep="0.005"/>

      <visual>
        <global offwidth="640" offheight="480"/>
      </visual>

      <asset>
        <mesh name="cube" scale="0.02 0.02 0.02" vertex="
          -1 -1 -1
           1 -1 -1
           1  1 -1
          -1  1 -1
          -1 -1  1
           1 -1  1
           1  1  1
          -1  1  1"/>
      </asset>

      <worldbody>
        <light pos="0.5 0 1.5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1"/>

        <!-- Camera -->
        <camera name="main_cam" pos="0.5 -0.8 0.6"
                xyaxes="1 0 0 0 0.6 0.8" fovy="45"/>

        <!-- Simple robot arm (simplified YAM) -->
        <body name="base" pos="0 0 0.1">
          <geom name="base_geom" type="cylinder" size="0.05 0.05"
                rgba="0.3 0.3 0.3 1"/>

          <body name="link1" pos="0 0 0.1">
            <joint name="joint1" type="hinge" axis="0 0 1" limited="false"/>
            <geom name="link1_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.15"
                  rgba="0.7 0.7 0.7 1"/>

            <body name="link2" pos="0 0 0.15">
              <joint name="joint2" type="hinge" axis="0 1 0" limited="false"/>
              <geom name="link2_geom" type="capsule" size="0.025" fromto="0 0 0 0.2 0 0"
                    rgba="0.6 0.6 0.6 1"/>

              <body name="link3" pos="0.2 0 0">
                <joint name="joint3" type="hinge" axis="0 1 0" limited="false"/>
                <geom name="link3_geom" type="capsule" size="0.02" fromto="0 0 0 0.15 0 0"
                      rgba="0.5 0.5 0.5 1"/>

                <!-- Gripper -->
                <body name="gripper" pos="0.15 0 0">
                  <geom name="gripper_palm" type="box" size="0.02 0.03 0.01"
                        rgba="0.4 0.4 0.4 1"/>

                  <body name="left_finger" pos="-0.025 0 0">
                    <joint name="left_finger_joint" type="slide" axis="1 0 0" limited="true" range="-0.03 0"/>
                    <geom name="left_finger_geom" type="box" size="0.005 0.02 0.02"
                          rgba="0.5 0.5 0.5 1"/>
                  </body>

                  <body name="right_finger" pos="0.025 0 0">
                    <joint name="right_finger_joint" type="slide" axis="1 0 0" limited="true" range="0 0.03"/>
                    <geom name="right_finger_geom" type="box" size="0.005 0.02 0.02"
                          rgba="0.5 0.5 0.5 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- Red cube to manipulate -->
        <body name="cube" pos="0.3 0.1 0.11">
          <joint type="free"/>
          <geom name="cube_geom" type="box" size="0.02 0.02 0.02"
                rgba="0.9 0.2 0.1 1" mass="0.05"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1" gear="10"/>
        <motor name="motor2" joint="joint2" gear="10"/>
        <motor name="motor3" joint="joint3" gear="10"/>
        <motor name="left_finger_motor" joint="left_finger_joint" gear="5"/>
        <motor name="right_finger_motor" joint="right_finger_joint" gear="5"/>
      </actuator>
    </mujoco>
    """

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(yam_scene_xml)
        scene_path = f.name

    print(f"✅ Created scene: {scene_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Set initial joint positions (reaching pose)
    data.qpos[0] = 0.5  # joint1
    data.qpos[1] = 0.8  # joint2
    data.qpos[2] = -0.5  # joint3
    data.qpos[3] = -0.01  # left finger
    data.qpos[4] = 0.01  # right finger

    mujoco.mj_forward(model, data)

    # Paths
    ply_path = Path(__file__).parent.parent / "data" / "pretrained" / "kitchen" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    output_dir = Path(__file__).parent.parent / "outputs" / "yam_standalone_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        print(f"❌ Background not found: {ply_path}")
        return

    print(f"✅ Background: {ply_path}")

    # Robot geoms for foreground
    robot_geoms = [
        'base_geom', 'link1_geom', 'link2_geom', 'link3_geom',
        'gripper_palm', 'left_finger_geom', 'right_finger_geom',
        'cube_geom',
    ]

    print(f"✅ Foreground geoms: {len(robot_geoms)}")

    # Create sensor
    print("\nCreating GaussianSensor...")
    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=ply_path,
        render_mode="hybrid",
        robot_geom_names=robot_geoms,
    )
    sensor = GaussianSensor(config)
    print("✅ Sensor created")

    # Render
    print("\nRendering...")
    result = sensor.render(model, data, "main_cam", return_components=True)

    print(f"✅ Hybrid shape: {result['rgb'].shape}")

    # Visualize
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(result['foreground'])
    ax1.set_title('MuJoCo Only', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(result['background'])
    ax2.set_title('3DGS Only', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(result['rgb'])
    ax3.set_title('MuGS Hybrid', fontsize=14, fontweight='bold', color='#d62728')
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(result['mask'], cmap='gray')
    ax4.set_title('Robot Mask', fontsize=12)
    ax4.axis('off')

    ax5 = plt.subplot(2, 3, 5)
    overlay = result['background'].copy().astype(float) / 255
    mask = result['mask'].squeeze()
    overlay[mask > 0.5] = [1, 0, 0]
    ax5.imshow(overlay)
    ax5.set_title('Robot Highlight', fontsize=12)
    ax5.axis('off')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    robot_ratio = mask.mean()
    stats = f"""
YAM + MuGS Demo

Robot: Simplified YAM arm
Object: Red cube (2cm)

Resolution: {result['rgb'].shape[1]}x{result['rgb'].shape[0]}
Robot pixels: {robot_ratio*100:.2f}%
Background: {(1-robot_ratio)*100:.2f}%

Mode: Hybrid rendering
"""
    ax6.text(0.1, 0.5, stats, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('YAM Manipulation + MuGS: Photorealistic Background',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "yam_comparison.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Copy to user outputs
    import shutil
    shutil.copy(output_path, "/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250/yam_mugs_demo.jpg")
    print("✅ Copied to outputs")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
