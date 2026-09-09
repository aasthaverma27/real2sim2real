"""
GaussianSensor Visible Robot Demo

Positions the robot within the pretrained camera's field of view to demonstrate
proper hybrid rendering with both photorealistic background and physics-simulated robot.

Author: MuGS Team
Date: 2026-05-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import mujoco
from PIL import Image

from mugs.sensors.gaussian_sensor import GaussianSensor, GaussianSensorConfig


# Robot positioned near the kitchen table where camera 100 can see it
ROBOT_SCENE_XML = """
<mujoco model="kitchen_robot_visible">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <!-- Robot positioned on/near the table visible to camera 100
         Camera 100 position: [2.93, 1.66, -1.08]
         Robot positioned slightly in front of camera view -->
    <body name="robot_base" pos="2.5 1.0 -0.5">
      <geom name="base_link" type="cylinder" size="0.05 0.03" rgba="1.0 0.1 0.1 1"/>

      <body name="shoulder" pos="0 0 0.03">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="shoulder_link" type="box" size="0.03 0.03 0.05" rgba="1.0 0.3 0.1 1"/>

        <body name="arm" pos="0 0 0.08" quat="0.707 0 0.707 0">
          <joint name="shoulder_lift" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom name="arm_link" type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" rgba="1.0 0.5 0.1 1"/>

          <body name="forearm" pos="0.2 0 0">
            <joint name="elbow" type="hinge" axis="0 0 1" range="-2.0 2.0"/>
            <geom name="forearm_link" type="capsule" size="0.018" fromto="0 0 0 0.15 0 0" rgba="1.0 0.7 0.1 1"/>

            <body name="gripper" pos="0.15 0 0">
              <joint name="wrist" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom name="palm" type="box" size="0.03 0.02 0.015" rgba="1.0 0.9 0.1 1"/>

              <body name="left_finger" pos="0.03 0.015 0">
                <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.03"/>
                <geom name="left_finger_link" type="box" size="0.006 0.01 0.01"
                      pos="0 0.01 0" rgba="1.0 0.9 0.1 1"/>
              </body>

              <body name="right_finger" pos="0.03 -0.015 0">
                <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.03"/>
                <geom name="right_finger_link" type="box" size="0.006 0.01 0.01"
                      pos="0 -0.01 0" rgba="1.0 0.9 0.1 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="dummy_cam" pos="1 1 1" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def main():
    """Demo hybrid rendering with visible robot."""

    print("=" * 70)
    print("GaussianSensor Visible Robot Demo")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs/gaussian_sensor_visible_robot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained camera
    print("\n[1/4] Loading pretrained camera...")
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"

    with open(cameras_path) as f:
        cameras = json.load(f)

    kitchen_camera = cameras[100]
    print(f"  ✓ Camera 100: {kitchen_camera['img_name']}")

    # Configure sensor
    print("\n[2/4] Configuring GaussianSensor...")

    config = GaussianSensorConfig(
        width=960,
        height=640,
        background_ply_path=base_dir / "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply",
        render_mode="hybrid",
        robot_geom_names=[
            'base_link', 'shoulder_link', 'arm_link', 'forearm_link',
            'palm', 'left_finger_link', 'right_finger_link'
        ],
        cache_background=True,
        device="cuda"
    )

    sensor = GaussianSensor(config)

    # Create MuJoCo scene
    print("\n[3/4] Creating MuJoCo scene...")

    model = mujoco.MjModel.from_xml_string(ROBOT_SCENE_XML)
    data = mujoco.MjData(model)

    print(f"  ✓ Robot positioned at [2.5, 1.0, -0.5]")

    # Render different robot poses
    print("\n[4/4] Rendering poses...")

    poses = [
        ("rest", np.array([0.0, 0.0, 0.0, 0.0, 0.005, 0.005])),
        ("reach_right", np.array([1.2, -0.5, 0.8, 0.0, 0.005, 0.005])),
        ("reach_up", np.array([0.0, -0.8, 1.0, -0.3, 0.005, 0.005])),
        ("extended", np.array([0.8, -1.0, 1.5, -0.5, 0.005, 0.005])),
        ("grasp", np.array([0.0, -0.3, 0.6, 0.0, 0.025, 0.025])),
    ]

    pose_images = []
    for pose_name, qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        result = sensor.render(model, data, "dummy_cam",
                              return_components=True,
                              camera_params=kitchen_camera)

        Image.fromarray(result['rgb']).save(output_dir / f"{pose_name}_hybrid.jpg", quality=95)
        Image.fromarray(result['foreground']).save(output_dir / f"{pose_name}_foreground.jpg", quality=90)
        pose_images.append(result['rgb'])

        robot_pixels = result['mask'].sum()
        total_pixels = result['mask'].size
        print(f"  ✓ {pose_name}: robot={robot_pixels:,}/{total_pixels:,} ({100*robot_pixels/total_pixels:.2f}%)")

    # Save reference background
    sensor.cfg.render_mode = "3dgs_only"
    bg = sensor.render(model, data, "dummy_cam", camera_params=kitchen_camera)
    Image.fromarray(bg).save(output_dir / "background_only.jpg", quality=95)
    sensor.cfg.render_mode = "hybrid"

    # Create comparison grid (3×2)
    print("  → Creating comparison grid...", end=' ')
    row1 = np.hstack([pose_images[0], pose_images[1]])
    row2 = np.hstack([pose_images[2], pose_images[3]])
    row3 = np.hstack([pose_images[4], bg])
    grid = np.vstack([row1, row2, row3])
    Image.fromarray(grid).save(output_dir / "poses_grid_3x2.jpg", quality=90)
    print("✓")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nOutputs: {output_dir}/")
    print(f"  • *_hybrid.jpg - 5 robot poses with kitchen background")
    print(f"  • *_foreground.jpg - Robot foreground renders")
    print(f"  • background_only.jpg - Clean kitchen reference")
    print(f"  • poses_grid_3x2.jpg - 3×2 comparison grid")
    print(f"\n✨ Hybrid rendering: photorealistic kitchen + visible robot!")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
