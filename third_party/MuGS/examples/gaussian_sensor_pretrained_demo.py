"""
GaussianSensor with Pretrained Camera Demo

Demonstrates using GaussianSensor with pretrained camera parameters from cameras.json
to properly align the 3DGS background with the MuJoCo scene.

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


# Simplified MuJoCo scene (robot only, no camera - we'll use pretrained camera)
ROBOT_SCENE_XML = """
<mujoco model="kitchen_robot_minimal">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <!-- Robot positioned to be visible from pretrained camera 100 viewpoint -->
    <body name="robot_base" pos="0 0 0.75">
      <geom name="base_link" type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.35 1"/>

      <body name="shoulder" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="shoulder_link" type="box" size="0.04 0.04 0.08" rgba="0.8 0.3 0.2 1"/>

        <body name="arm" pos="0 0 0.12" quat="0.707 0 0.707 0">
          <joint name="shoulder_lift" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom name="arm_link" type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" rgba="0.2 0.6 0.8 1"/>

          <body name="forearm" pos="0.3 0 0">
            <joint name="elbow" type="hinge" axis="0 0 1" range="-2.0 2.0"/>
            <geom name="forearm_link" type="capsule" size="0.025" fromto="0 0 0 0.25 0 0" rgba="0.2 0.8 0.6 1"/>

            <body name="gripper" pos="0.25 0 0">
              <joint name="wrist" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom name="palm" type="box" size="0.04 0.03 0.02" rgba="0.9 0.7 0.1 1"/>

              <body name="left_finger" pos="0.04 0.02 0">
                <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.04"/>
                <geom name="left_finger_link" type="box" size="0.008 0.015 0.015"
                      pos="0 0.015 0" rgba="0.9 0.7 0.1 1"/>
              </body>

              <body name="right_finger" pos="0.04 -0.02 0">
                <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.04"/>
                <geom name="right_finger_link" type="box" size="0.008 0.015 0.015"
                      pos="0 -0.015 0" rgba="0.9 0.7 0.1 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Dummy camera (required for MuJoCo rendering, but we'll use pretrained params for 3DGS) -->
    <camera name="dummy_cam" pos="1 1 1" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def main():
    """Demo GaussianSensor with pretrained camera parameters."""

    print("=" * 70)
    print("GaussianSensor with Pretrained Camera Demo")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs/gaussian_sensor_pretrained_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained camera parameters
    print("\n[1/4] Loading pretrained camera...")
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"

    with open(cameras_path) as f:
        cameras = json.load(f)

    # Use camera 100 (good kitchen view)
    kitchen_camera = cameras[100]
    print(f"  ✓ Loaded camera {kitchen_camera['id']}: {kitchen_camera['img_name']}")
    print(f"    Position: [{kitchen_camera['position'][0]:.2f}, {kitchen_camera['position'][1]:.2f}, {kitchen_camera['position'][2]:.2f}]")
    print(f"    Resolution: {kitchen_camera['width']}×{kitchen_camera['height']}")

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

    # Set robot pose
    data.qpos[0] = 0.5   # shoulder_pan
    data.qpos[1] = -0.8  # shoulder_lift
    data.qpos[2] = 1.2   # elbow
    data.qpos[3] = -0.5  # wrist
    data.qpos[4] = 0.02  # left_finger
    data.qpos[5] = 0.02  # right_finger

    mujoco.mj_forward(model, data)

    print(f"  ✓ Robot: {model.nq} DOF")

    # Render with pretrained camera
    print("\n[4/4] Rendering...")

    print("  → Hybrid rendering with pretrained camera...", end=' ')
    hybrid_result = sensor.render(
        model, data, "dummy_cam",
        return_components=True,
        camera_params=kitchen_camera  # Use pretrained camera for 3DGS background!
    )
    Image.fromarray(hybrid_result['rgb']).save(output_dir / "hybrid_pretrained_cam.jpg", quality=95)
    Image.fromarray(hybrid_result['background']).save(output_dir / "background_pretrained_cam.jpg", quality=95)
    print("✓")

    # Also render with just 3DGS (no robot) to show clean background
    print("  → 3DGS only...", end=' ')
    sensor.cfg.render_mode = "3dgs_only"
    rgb_3dgs = sensor.render(model, data, "dummy_cam", camera_params=kitchen_camera)
    Image.fromarray(rgb_3dgs).save(output_dir / "kitchen_3dgs_only.jpg", quality=95)
    print("✓")

    # Back to hybrid
    sensor.cfg.render_mode = "hybrid"

    # Test different robot poses with same camera
    print("  → Rendering multiple poses...", end=' ')
    poses = [
        ("rest", np.array([0.0, -0.3, 0.5, -0.2, 0.01, 0.01])),
        ("reach", np.array([0.8, -0.6, 1.0, -0.4, 0.01, 0.01])),
        ("extended", np.array([0.8, -0.8, 1.4, -0.6, 0.01, 0.01])),
        ("grasp", np.array([0.8, -0.8, 1.4, -0.6, 0.005, 0.005])),
    ]

    pose_images = []
    for pose_name, qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        rgb = sensor.render(model, data, "dummy_cam", camera_params=kitchen_camera)
        Image.fromarray(rgb).save(output_dir / f"pose_{pose_name}.jpg", quality=90)
        pose_images.append(rgb)

    print("✓")

    # Create comparison grid
    print("  → Creating comparison grid...", end=' ')
    top_row = np.hstack([pose_images[0], pose_images[1]])
    bottom_row = np.hstack([pose_images[2], pose_images[3]])
    grid = np.vstack([top_row, bottom_row])
    Image.fromarray(grid).save(output_dir / "poses_grid_2x2.jpg", quality=90)
    print("✓")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nOutputs: {output_dir}/")
    print(f"  • hybrid_pretrained_cam.jpg - Hybrid render with pretrained camera")
    print(f"  • background_pretrained_cam.jpg - 3DGS kitchen background")
    print(f"  • kitchen_3dgs_only.jpg - Clean kitchen view (no robot)")
    print(f"  • pose_*.jpg - 4 different robot poses")
    print(f"  • poses_grid_2x2.jpg - 2×2 grid comparison")
    print(f"\n✨ Photorealistic kitchen background now renders correctly!")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
