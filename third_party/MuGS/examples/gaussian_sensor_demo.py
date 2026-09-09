"""
GaussianSensor Usage Demo

Demonstrates the new unified GaussianSensor API for hybrid rendering.

Author: MuGS Team
Date: 2026-05-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mujoco
from PIL import Image

from mugs.sensors.gaussian_sensor import GaussianSensor, GaussianSensorConfig


# MuJoCo scene with robot
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

    <camera name="kitchen_cam" mode="targetbody" target="robot_base"
            pos="0.6 -0.8 1.2"/>
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
    """Demo GaussianSensor API."""

    print("=" * 70)
    print("GaussianSensor API Demo")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs/gaussian_sensor_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Configure sensor
    print("\n[1/4] Configuring GaussianSensor...")

    config = GaussianSensorConfig(
        width=960,
        height=640,
        background_ply_path=base_dir / "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply",
        render_mode="hybrid",  # 'hybrid', '3dgs_only', or 'mujoco_only'
        robot_geom_names=[
            'base_link', 'shoulder_link', 'arm_link', 'forearm_link',
            'palm', 'left_finger_link', 'right_finger_link'
        ],
        cache_background=True,  # Cache static 3DGS background
        device="cuda"
    )

    sensor = GaussianSensor(config)

    print(f"\n  Sensor stats:")
    for key, val in sensor.get_stats().items():
        print(f"    {key}: {val}")

    # 2. Create MuJoCo environment
    print("\n[2/4] Creating MuJoCo environment...")

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

    # 3. Render with different modes
    print("\n[3/4] Rendering...")

    # Hybrid mode (default)
    print("  → Hybrid rendering (3DGS + MuJoCo)...", end=' ')
    hybrid_result = sensor.render(model, data, "kitchen_cam", return_components=True)
    Image.fromarray(hybrid_result['rgb']).save(output_dir / "hybrid.jpg", quality=95)
    Image.fromarray(hybrid_result['background']).save(output_dir / "background_3dgs.jpg", quality=95)
    Image.fromarray(hybrid_result['foreground']).save(output_dir / "foreground_mujoco.jpg", quality=95)
    Image.fromarray(hybrid_result['mask'] * 255).save(output_dir / "mask.png")
    print("✓")

    # 3DGS only
    print("  → 3DGS only...", end=' ')
    sensor.cfg.render_mode = "3dgs_only"
    rgb_3dgs = sensor.render(model, data, "kitchen_cam")
    Image.fromarray(rgb_3dgs).save(output_dir / "3dgs_only.jpg", quality=95)
    print("✓")

    # MuJoCo only
    print("  → MuJoCo only...", end=' ')
    sensor.cfg.render_mode = "mujoco_only"
    rgb_mujoco = sensor.render(model, data, "kitchen_cam")
    Image.fromarray(rgb_mujoco).save(output_dir / "mujoco_only.jpg", quality=95)
    print("✓")

    # Back to hybrid
    sensor.cfg.render_mode = "hybrid"

    # 4. Test cache performance
    print("\n[4/4] Testing background cache...")

    import time

    # Cold render (no cache)
    sensor.clear_cache()
    start = time.perf_counter()
    _ = sensor.render(model, data, "kitchen_cam")
    cold_time = (time.perf_counter() - start) * 1000

    # Warm render (cached background)
    start = time.perf_counter()
    _ = sensor.render(model, data, "kitchen_cam")
    warm_time = (time.perf_counter() - start) * 1000

    print(f"  Cold render: {cold_time:.2f} ms")
    print(f"  Warm render: {warm_time:.2f} ms")
    print(f"  Speedup: {cold_time / warm_time:.2f}×")

    # 5. Create comparison
    print("\n[5/5] Creating comparison...")

    comparison = np.hstack([
        hybrid_result['background'],
        hybrid_result['foreground'],
        hybrid_result['rgb']
    ])
    Image.fromarray(comparison).save(output_dir / "comparison.jpg", quality=90)
    print(f"  ✓ comparison.jpg")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nOutputs: {output_dir}/")
    print(f"  • hybrid.jpg - Final composite")
    print(f"  • background_3dgs.jpg - 3DGS background")
    print(f"  • foreground_mujoco.jpg - MuJoCo foreground")
    print(f"  • mask.png - Robot segmentation mask")
    print(f"  • 3dgs_only.jpg - 3DGS only mode")
    print(f"  • mujoco_only.jpg - MuJoCo only mode")
    print(f"  • comparison.jpg - Side-by-side (BG | FG | Hybrid)")
    print(f"\n✨ GaussianSensor API: Clean, unified, fast!")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
