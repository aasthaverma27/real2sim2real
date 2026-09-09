"""
GaussianSensor Complete Working Hybrid Demo

Uses the working approach from Phase 2 demos but through the GaussianSensor API.
- 3DGS background: pretrained camera (photorealistic kitchen view)
- MuJoCo foreground: trackbody camera (follows robot)
- Compositing: segmentation-based alpha blending

Note: Cameras have different viewpoints (not geometrically aligned). This is acceptable
for demonstration purposes. Future work: full coordinate transformation for spatial alignment.

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


# Same robot scene as original hybrid demo
ROBOT_SCENE_XML = """
<mujoco model="kitchen_robot">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.25 1"/>
    <light pos="1 1 3" dir="-1 -1 -2"/>

    <body name="robot_base" pos="0 0 0.75">
      <geom name="base_link" type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.35 1"/>

      <body name="shoulder" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
        <geom name="shoulder_link" type="box" size="0.04 0.04 0.08" rgba="0.8 0.3 0.2 1"/>

        <body name="arm" pos="0 0 0.12" quat="0.707 0 0.707 0">
          <joint name="shoulder_lift" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.3"/>
          <geom name="arm_link" type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" rgba="0.2 0.6 0.8 1"/>

          <body name="forearm" pos="0.3 0 0">
            <joint name="elbow" type="hinge" axis="0 0 1" range="-2.0 2.0" damping="0.2"/>
            <geom name="forearm_link" type="capsule" size="0.025" fromto="0 0 0 0.25 0 0" rgba="0.2 0.8 0.6 1"/>

            <body name="gripper" pos="0.25 0 0">
              <joint name="wrist" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.1"/>
              <geom name="palm" type="box" size="0.04 0.03 0.02" rgba="0.9 0.7 0.1 1"/>

              <body name="left_finger" pos="0.04 0.02 0">
                <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.04" damping="0.05"/>
                <geom name="left_finger_link" type="box" size="0.008 0.015 0.015"
                      pos="0 0.015 0" rgba="0.9 0.7 0.1 1"/>
              </body>

              <body name="right_finger" pos="0.04 -0.02 0">
                <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.04" damping="0.05"/>
                <geom name="right_finger_link" type="box" size="0.008 0.015 0.015"
                      pos="0 -0.015 0" rgba="0.9 0.7 0.1 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="kitchen_cam" mode="targetbody" target="robot_base" pos="0.6 -0.8 1.2"/>
  </worldbody>

  <actuator>
    <position name="act_shoulder_pan" joint="shoulder_pan" kp="20"/>
    <position name="act_shoulder_lift" joint="shoulder_lift" kp="20"/>
    <position name="act_elbow" joint="elbow" kp="15"/>
    <position name="act_wrist" joint="wrist" kp="10"/>
    <position name="act_left_finger" joint="left_finger_joint" kp="5"/>
    <position name="act_right_finger" joint="right_finger_joint" kp="5"/>
  </actuator>
</mujoco>
"""


def main():
    """Demo working hybrid rendering through GaussianSensor API."""

    print("=" * 70)
    print("GaussianSensor Complete Working Hybrid Demo")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs/gaussian_sensor_working_hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained camera for 3DGS background
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    kitchen_camera = cameras[100]  # Pretrained camera with kitchen view

    # Configure sensor
    print("\n[1/3] Configuring GaussianSensor...")
    print(f"  → 3DGS camera: {kitchen_camera['img_name']} (pretrained)")
    print(f"  → MuJoCo camera: kitchen_cam (trackbody)")

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

    print(f"  ✓ GaussianSensor ready")

    # Create MuJoCo scene
    print("\n[2/3] Creating MuJoCo scene...")

    model = mujoco.MjModel.from_xml_string(ROBOT_SCENE_XML)
    data = mujoco.MjData(model)

    print(f"  ✓ Robot: {model.nq} DOF")

    # Render different poses
    print("\n[3/3] Rendering poses...")

    poses = [
        ("rest", np.array([0.0, -0.3, 0.5, -0.2, 0.01, 0.01])),
        ("reach", np.array([0.8, -0.6, 1.0, -0.4, 0.01, 0.01])),
        ("extended", np.array([0.8, -0.8, 1.4, -0.6, 0.01, 0.01])),
        ("open_gripper", np.array([0.8, -0.8, 1.4, -0.6, 0.03, 0.03])),
        ("grasp", np.array([0.8, -0.8, 1.4, -0.6, 0.005, 0.005])),
        ("return", np.array([0.0, -0.3, 0.5, -0.2, 0.005, 0.005])),
    ]

    pose_images = []
    mask_stats = []

    for pose_name, qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        # Render through API (pretrained camera for 3DGS, MuJoCo camera for foreground)
        result = sensor.render(model, data, "kitchen_cam",
                              return_components=True,
                              camera_params=kitchen_camera)

        # Save outputs
        Image.fromarray(result['rgb']).save(output_dir / f"{pose_name}_hybrid.jpg", quality=95)
        Image.fromarray(result['background']).save(output_dir / f"{pose_name}_background.jpg", quality=90)
        Image.fromarray(result['foreground']).save(output_dir / f"{pose_name}_foreground.jpg", quality=90)
        Image.fromarray(result['mask'] * 255).save(output_dir / f"{pose_name}_mask.png")

        pose_images.append(result['rgb'])

        robot_pixels = result['mask'].sum()
        total_pixels = result['mask'].size
        pct = 100 * robot_pixels / total_pixels
        mask_stats.append((pose_name, robot_pixels, pct))

        print(f"  ✓ {pose_name}: {robot_pixels:,} pixels ({pct:.2f}%)")

    # Save reference renders
    print("  → Saving reference renders...", end=' ')
    sensor.cfg.render_mode = "3dgs_only"
    bg_only = sensor.render(model, data, "kitchen_cam", camera_params=kitchen_camera)
    Image.fromarray(bg_only).save(output_dir / "ref_3dgs_only.jpg", quality=95)

    sensor.cfg.render_mode = "mujoco_only"
    fg_only = sensor.render(model, data, "kitchen_cam")  # MuJoCo uses its own camera
    Image.fromarray(fg_only).save(output_dir / "ref_mujoco_only.jpg", quality=95)

    sensor.cfg.render_mode = "hybrid"
    print("✓")

    # Create comparison grids
    print("  → Creating comparison grids...", end=' ')

    # Grid 1: First 4 poses in 2×2
    grid1 = np.vstack([
        np.hstack([pose_images[0], pose_images[1]]),
        np.hstack([pose_images[2], pose_images[3]])
    ])
    Image.fromarray(grid1).save(output_dir / "poses_grid1_2x2.jpg", quality=90)

    # Grid 2: Last 2 poses + references
    grid2 = np.vstack([
        np.hstack([pose_images[4], pose_images[5]]),
        np.hstack([bg_only, fg_only])
    ])
    Image.fromarray(grid2).save(output_dir / "poses_grid2_2x2.jpg", quality=90)
    print("✓")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nOutputs: {output_dir}/")
    print(f"  • *_hybrid.jpg - 6 robot poses with 3DGS background")
    print(f"  • *_background/foreground/mask.* - Individual components")
    print(f"  • ref_*.jpg - 3DGS-only and MuJoCo-only references")
    print(f"  • poses_grid*.jpg - 2×2 comparison grids")
    print(f"\nMask coverage:")
    for name, pixels, pct in mask_stats:
        print(f"  • {name:15s}: {pixels:6,} pixels ({pct:5.2f}%)")
    print(f"\n✨ GaussianSensor API: Photorealistic + Physics-accurate = VLA-ready!")
    print(f"\nApproach:")
    print(f"  • 3DGS background: Pretrained camera (photorealistic kitchen)")
    print(f"  • MuJoCo foreground: Trackbody camera (follows robot)")
    print(f"  • Viewpoints differ but compositing works visually")
    print(f"  • Future: Full coordinate transforms for geometric alignment")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
