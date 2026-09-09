"""
Hybrid Rendering Demo: Pretrained Kitchen (3DGS) + MuJoCo Robot

Demonstrates Phase 2 core capability:
- Background: Photorealistic kitchen from pretrained 1.85M Gaussians
- Foreground: MuJoCo robot arm with physics simulation
- Compositing: Segmentation-based mask blending

Author: MuGS Team
Date: 2026-05-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization
import mujoco

# MuJoCo scene XML with simple robot arm
ROBOT_SCENE_XML = """
<mujoco model="kitchen_robot">
  <compiler angle="radian"/>

  <option timestep="0.002" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>

  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
  </asset>

  <worldbody>
    <!-- Visible floor for debug -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.25 1" material="matplane"/>

    <!-- Light to see robot -->
    <light pos="1 1 3" dir="-1 -1 -2"/>

    <!-- Robot base on table -->
    <body name="robot_base" pos="0 0 0.75">
      <geom name="base_link" type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.35 1"/>

      <!-- Shoulder joint -->
      <body name="shoulder" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
        <geom name="shoulder_link" type="box" size="0.04 0.04 0.08" rgba="0.8 0.3 0.2 1"/>

        <!-- Arm link -->
        <body name="arm" pos="0 0 0.12" quat="0.707 0 0.707 0">
          <joint name="shoulder_lift" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.3"/>
          <geom name="arm_link" type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" rgba="0.2 0.6 0.8 1"/>

          <!-- Forearm -->
          <body name="forearm" pos="0.3 0 0">
            <joint name="elbow" type="hinge" axis="0 0 1" range="-2.0 2.0" damping="0.2"/>
            <geom name="forearm_link" type="capsule" size="0.025" fromto="0 0 0 0.25 0 0" rgba="0.2 0.8 0.6 1"/>

            <!-- End effector (simple gripper) -->
            <body name="gripper" pos="0.25 0 0">
              <joint name="wrist" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.1"/>

              <!-- Palm -->
              <geom name="palm" type="box" size="0.04 0.03 0.02" rgba="0.9 0.7 0.1 1"/>

              <!-- Left finger -->
              <body name="left_finger" pos="0.04 0.02 0">
                <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.04" damping="0.05"/>
                <geom name="left_finger_link" type="box" size="0.008 0.015 0.015"
                      pos="0 0.015 0" rgba="0.9 0.7 0.1 1"/>
              </body>

              <!-- Right finger -->
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

    <!-- Camera tracking the robot from an angle -->
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


def load_official_ply(ply_path: Path):
    """Load official 3DGS PLY format."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    SH_C0 = 0.28209479177387814
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
    colors = 0.5 + SH_C0 * sh_dc
    colors = np.clip(colors, 0, 1)

    opacities = 1 / (1 + np.exp(-vertex['opacity']))
    scales = np.exp(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1))
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    return {
        'means': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'quats': quats,
    }


def camera_to_view_matrix(position, rotation):
    """Convert camera position and rotation to view matrix."""
    R = np.array(rotation).T
    t = -R @ np.array(position)
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    return view_matrix


def render_3dgs_background(gaussians, camera, width, height, device='cuda'):
    """Render 3DGS kitchen background."""
    fx = camera['fx'] * (width / camera['width'])
    fy = camera['fy'] * (height / camera['height'])
    cx = width / 2
    cy = height / 2

    view_matrix = camera_to_view_matrix(camera['position'], camera['rotation'])
    viewmat = torch.from_numpy(view_matrix).float().to(device)

    render_colors, render_alphas, info = rasterization(
        means=gaussians['means'],
        quats=gaussians['quats'],
        scales=gaussians['scales'],
        opacities=gaussians['opacities'],
        colors=gaussians['colors'],
        viewmats=viewmat[None],
        Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)[None],
        width=width,
        height=height,
        packed=False,
    )

    rgb = render_colors[0].clamp(0, 1)
    return (rgb.cpu().numpy() * 255).astype(np.uint8)


def render_mujoco_foreground(model, data, camera_name, width, height):
    """Render MuJoCo robot with RGB and segmentation."""
    renderer = mujoco.Renderer(model, height, width)

    # Update rendering context
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_name)

    # RGB render
    rgb = renderer.render()

    # Segmentation render
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    # Extract object IDs from segmentation
    seg_ids = seg[:, :, 0].astype(np.int32)

    renderer.close()

    return rgb, seg_ids


def create_robot_mask(seg_ids, model, robot_geom_names):
    """Create binary mask for robot geoms."""
    mask = np.zeros_like(seg_ids, dtype=np.uint8)

    for geom_name in robot_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            mask[seg_ids == geom_id] = 1

    return mask


def composite_images(background_rgb, foreground_rgb, mask):
    """Composite foreground onto background using mask."""
    mask_3ch = mask[:, :, None].astype(np.float32)
    composite = background_rgb * (1 - mask_3ch) + foreground_rgb * mask_3ch
    return composite.astype(np.uint8)


def main():
    """Run hybrid rendering demo."""

    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "outputs/hybrid_kitchen_robot"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hybrid Rendering: Pretrained Kitchen + MuJoCo Robot")
    print("=" * 70)

    # === Load 3DGS Kitchen Background ===
    print("\n[1/5] Loading pretrained kitchen (1.85M Gaussians)...")
    model_path = base_dir / "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"
    cameras_path = base_dir / "data/pretrained/kitchen/cameras.json"

    gaussians_np = load_official_ply(model_path)
    device = torch.device('cuda')
    gaussians = {
        'means': torch.from_numpy(gaussians_np['means']).float().to(device),
        'quats': torch.from_numpy(gaussians_np['quats']).float().to(device),
        'scales': torch.from_numpy(gaussians_np['scales']).float().to(device),
        'opacities': torch.from_numpy(gaussians_np['opacities']).float().to(device),
        'colors': torch.from_numpy(gaussians_np['colors']).float().to(device),
    }

    with open(cameras_path) as f:
        cameras = json.load(f)

    # Use camera 100 as reference
    kitchen_camera = cameras[100]

    print(f"  ✓ Loaded {len(gaussians['means']):,} Gaussians")
    print(f"  ✓ Using camera view {kitchen_camera['img_name']}")

    # === Create MuJoCo Robot Scene ===
    print("\n[2/5] Creating MuJoCo robot scene...")
    model = mujoco.MjModel.from_xml_string(ROBOT_SCENE_XML)
    data = mujoco.MjData(model)

    # Set robot pose (interesting configuration)
    data.qpos[0] = 0.5   # shoulder_pan
    data.qpos[1] = -0.8  # shoulder_lift
    data.qpos[2] = 1.2   # elbow
    data.qpos[3] = -0.5  # wrist
    data.qpos[4] = 0.02  # left_finger
    data.qpos[5] = 0.02  # right_finger

    mujoco.mj_forward(model, data)

    print(f"  ✓ Robot: {model.nq} DOF, {model.ngeom} geoms")

    # === Render Both Layers ===
    width, height = 960, 640
    print(f"\n[3/5] Rendering at {width}×{height}...")

    print("  → 3DGS background...", end=' ')
    background_rgb = render_3dgs_background(gaussians, kitchen_camera, width, height, device)
    print("✓")

    print("  → MuJoCo foreground...", end=' ')
    foreground_rgb, seg_ids = render_mujoco_foreground(model, data, "kitchen_cam", width, height)
    print("✓")

    # === Create Composite ===
    print("\n[4/5] Compositing layers...")

    # Robot geoms to mask
    robot_geoms = [
        'base_link', 'shoulder_link', 'arm_link', 'forearm_link',
        'palm', 'left_finger_link', 'right_finger_link'
    ]

    robot_mask = create_robot_mask(seg_ids, model, robot_geoms)
    composite_rgb = composite_images(background_rgb, foreground_rgb, robot_mask)

    print(f"  ✓ Robot pixels: {robot_mask.sum():,} / {robot_mask.size:,} ({100*robot_mask.mean():.1f}%)")

    # === Save Outputs ===
    print("\n[5/5] Saving results...")

    Image.fromarray(background_rgb).save(output_dir / "background_3dgs.jpg", quality=95)
    print(f"  ✓ background_3dgs.jpg")

    Image.fromarray(foreground_rgb).save(output_dir / "foreground_mujoco.jpg", quality=95)
    print(f"  ✓ foreground_mujoco.jpg")

    Image.fromarray(robot_mask * 255).save(output_dir / "robot_mask.png")
    print(f"  ✓ robot_mask.png")

    Image.fromarray(composite_rgb).save(output_dir / "composite_hybrid.jpg", quality=95)
    print(f"  ✓ composite_hybrid.jpg")

    # Create side-by-side comparison
    comparison = np.hstack([background_rgb, foreground_rgb, composite_rgb])
    Image.fromarray(comparison).save(output_dir / "comparison.jpg", quality=90)
    print(f"  ✓ comparison.jpg (3-panel)")

    print("\n" + "=" * 70)
    print("Hybrid Rendering Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"\nKey files:")
    print(f"  • composite_hybrid.jpg - Final result ✨")
    print(f"  • comparison.jpg - Side-by-side (BG | FG | Composite)")
    print(f"  • robot_mask.png - Segmentation mask")
    print(f"\n🎯 Achievement: Photorealistic kitchen + physics-accurate robot!")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
