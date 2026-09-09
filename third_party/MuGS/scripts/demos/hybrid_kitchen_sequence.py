"""
Hybrid Rendering Sequence: Animated Robot in Photorealistic Kitchen

Demonstrates dynamic hybrid rendering with robot motion:
- Multiple robot poses (reaching, grasping, retracting)
- Consistent 3DGS background
- Real-time compositing at each frame

Author: MuGS Team
Date: 2026-05-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from gsplat import rasterization
import mujoco

# Import robot scene from previous demo
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_kitchen_robot import (
    ROBOT_SCENE_XML,
    load_official_ply,
    camera_to_view_matrix,
    render_3dgs_background,
    render_mujoco_foreground,
    create_robot_mask,
    composite_images
)


def interpolate_poses(pose_start, pose_end, steps):
    """Linear interpolation between two poses."""
    poses = []
    for i in range(steps):
        alpha = i / (steps - 1)
        pose = pose_start * (1 - alpha) + pose_end * alpha
        poses.append(pose)
    return poses


def main():
    """Generate hybrid rendering sequence."""

    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "outputs/hybrid_kitchen_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hybrid Rendering Sequence: Robot Motion in Kitchen")
    print("=" * 70)

    # Load 3DGS kitchen
    print("\n[1/4] Loading kitchen background...")
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
    kitchen_camera = cameras[100]

    print(f"  ✓ Kitchen: {len(gaussians['means']):,} Gaussians")

    # Create MuJoCo robot
    print("\n[2/4] Creating robot scene...")
    model = mujoco.MjModel.from_xml_string(ROBOT_SCENE_XML)
    data = mujoco.MjData(model)

    # Define robot motion sequence (6 keyframes)
    poses_key = [
        np.array([0.0, -0.3, 0.5, -0.2, 0.01, 0.01]),  # Rest
        np.array([0.8, -0.6, 1.0, -0.4, 0.01, 0.01]),  # Reach out
        np.array([0.8, -0.8, 1.4, -0.6, 0.01, 0.01]),  # Reach further
        np.array([0.8, -0.8, 1.4, -0.6, 0.03, 0.03]),  # Open gripper
        np.array([0.8, -0.8, 1.4, -0.6, 0.005, 0.005]), # Close gripper (grasp)
        np.array([0.0, -0.3, 0.5, -0.2, 0.005, 0.005]), # Return home
    ]

    # Interpolate between keyframes
    steps_per_transition = 8
    all_poses = []
    for i in range(len(poses_key) - 1):
        interp = interpolate_poses(poses_key[i], poses_key[i+1], steps_per_transition)
        all_poses.extend(interp[:-1])  # Avoid duplicates
    all_poses.append(poses_key[-1])

    print(f"  ✓ Motion: {len(all_poses)} frames")

    # Render sequence
    width, height = 960, 640
    print(f"\n[3/4] Rendering {len(all_poses)} frames at {width}×{height}...")

    robot_geoms = [
        'base_link', 'shoulder_link', 'arm_link', 'forearm_link',
        'palm', 'left_finger_link', 'right_finger_link'
    ]

    # Render background once (static)
    print("  → Rendering 3DGS background...", end=' ')
    background_rgb = render_3dgs_background(gaussians, kitchen_camera, width, height, device)
    print("✓")

    frames = []
    for idx, pose in enumerate(all_poses):
        # Set robot pose
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)

        # Render MuJoCo foreground
        foreground_rgb, seg_ids = render_mujoco_foreground(model, data, "kitchen_cam", width, height)

        # Create composite
        robot_mask = create_robot_mask(seg_ids, model, robot_geoms)
        composite_rgb = composite_images(background_rgb, foreground_rgb, robot_mask)

        # Add frame number text
        img = Image.fromarray(composite_rgb)
        draw = ImageDraw.Draw(img)
        text = f"Frame {idx+1}/{len(all_poses)}"
        draw.text((10, 10), text, fill=(255, 255, 0))

        frames.append(np.array(img))

        if (idx + 1) % 10 == 0:
            print(f"  ✓ Rendered {idx+1}/{len(all_poses)} frames")

    print(f"  ✓ All {len(frames)} frames complete")

    # Save sequence
    print(f"\n[4/4] Saving results...")

    # Save individual frames
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(output_dir / f"frame_{idx:03d}.jpg", quality=90)
    print(f"  ✓ Saved {len(frames)} individual frames")

    # Save key moments
    key_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    key_grid = np.vstack([
        np.hstack([frames[key_indices[0]], frames[key_indices[1]]]),
        np.hstack([frames[key_indices[2]], frames[key_indices[3]]]),
    ])
    # Add last frame to complete 2x3 grid
    last_row = np.hstack([frames[key_indices[4]], np.zeros_like(frames[0])])
    key_grid = np.vstack([key_grid, last_row])

    Image.fromarray(key_grid[:, :width*2, :]).save(output_dir / "key_moments_2x2.jpg", quality=90)
    print(f"  ✓ key_moments_2x2.jpg")

    # Create animated GIF (sample every 2nd frame for smaller file)
    gif_frames = [Image.fromarray(f) for f in frames[::2]]
    gif_frames[0].save(
        output_dir / "robot_motion.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,  # ms per frame
        loop=0
    )
    print(f"  ✓ robot_motion.gif ({len(gif_frames)} frames)")

    print("\n" + "=" * 70)
    print("Sequence Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  • {len(frames)} individual frames: frame_*.jpg")
    print(f"  • Key moments grid: key_moments_2x2.jpg")
    print(f"  • Animation: robot_motion.gif")
    print(f"\n🎬 Hybrid rendering: photorealistic background + dynamic robot!")


if __name__ == "__main__":
    import os
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    main()
