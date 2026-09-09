#!/usr/bin/env python3
"""
MuGS + mjlab YAM Lift Cube Demo

Test GaussianSensor hybrid rendering on a real manipulation RL task.
Shows comparison between:
- MuJoCo only (baseline)
- 3DGS only (background)
- MuGS Hybrid (photorealistic + physics)
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/ununtu/code/glab/mjlab/src")

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import mjlab task
from mjlab.tasks.manipulation.config.yam import yam_lift_cube_env_cfg
from mjlab.envs import ManagerBasedRlEnv

# Import MuGS sensor
try:
    from mugs.sensors.gaussian_sensor_mjlab import GaussianSensorMjlabCfg
except ImportError:
    print("❌ MuGS mjlab support not available")
    print("   Please ensure mjlab is installed")
    sys.exit(1)


def main():
    print("=" * 80)
    print("MuGS + mjlab YAM Lift Cube Demo")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Paths
    ply_path = Path(__file__).parent.parent / "data" / "pretrained" / "kitchen" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    output_dir = Path(__file__).parent.parent / "outputs" / "mjlab_yam_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        print(f"❌ Background model not found: {ply_path}")
        print("   Run: python scripts/download_pretrained_models.py")
        return

    print(f"✅ Background: {ply_path}")

    # Create environment
    print("\n" + "=" * 80)
    print("Creating YAM Lift Cube environment...")
    print("=" * 80)

    env_cfg = yam_lift_cube_env_cfg(play=True)  # Play mode for visualization
    env_cfg.scene.num_envs = 4  # Small batch for demo

    # Add GaussianSensor to the environment
    # Get all YAM geoms for foreground rendering
    yam_geoms = [
        # Robot links
        "arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
        # Gripper
        r"lf_down.*", r"rf_down.*",
        # Cube
        "cube_geom",
    ]

    gaussian_sensor_cfg = GaussianSensorMjlabCfg(
        name="gaussian_rgb",
        camera_name="robot/camera_d405",  # Use YAM's built-in camera
        width=640,
        height=480,
        background_ply_path=str(ply_path),
        render_mode="hybrid",
        robot_geom_names=yam_geoms,
        cache_background=True,
        return_components=True,
    )

    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (gaussian_sensor_cfg,)

    print("Creating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    print(f"✅ Environment created")
    print(f"   Num envs: {env.num_envs}")
    print(f"   Episode length: {env.max_episode_length}")

    # Reset and run a few steps
    print("\n" + "=" * 80)
    print("Running simulation...")
    print("=" * 80)

    obs_dict, _ = env.reset()

    # Take random actions for a few steps to see motion
    for step in range(50):
        action = torch.randn(env.num_envs, env.action_space.shape[0], device=device) * 0.1
        obs_dict, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(f"  Step {step}: reward={reward[0].item():.3f}")

    # Get rendering results
    print("\n" + "=" * 80)
    print("Extracting rendering results...")
    print("=" * 80)

    if "gaussian_rgb" in obs_dict:
        gaussian_obs = obs_dict["gaussian_rgb"]

        # Extract first environment's results
        results = {
            'hybrid': gaussian_obs.rgb[0].cpu().numpy(),
            'background': gaussian_obs.background[0].cpu().numpy(),
            'foreground': gaussian_obs.foreground[0].cpu().numpy(),
            'mask': gaussian_obs.mask[0].cpu().numpy(),
        }

        print(f"✅ Hybrid RGB shape: {results['hybrid'].shape}")
        print(f"✅ Background shape: {results['background'].shape}")
        print(f"✅ Foreground shape: {results['foreground'].shape}")
        print(f"✅ Mask shape: {results['mask'].shape}")

        # Visualization
        print("\n" + "=" * 80)
        print("Generating visualization...")
        print("=" * 80)

        fig = plt.figure(figsize=(20, 10))

        # Row 1: Main comparison
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(results['foreground'])
        ax1.set_title('MuJoCo Only\n(Crude Simulation)', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(results['background'])
        ax2.set_title('3DGS Only\n(Photorealistic)', fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(results['hybrid'])
        ax3.set_title('MuGS Hybrid\n(Best of Both Worlds)', fontsize=12, fontweight='bold', color='#d62728')
        ax3.axis('off')

        ax4 = plt.subplot(2, 4, 4)
        ax4.imshow(results['mask'], cmap='gray')
        ax4.set_title('Robot Mask\n(Foreground Region)', fontsize=12)
        ax4.axis('off')

        # Row 2: Detailed views
        ax5 = plt.subplot(2, 4, 5)
        # Overlay mask on background
        overlay = results['background'].copy().astype(float) / 255
        mask_alpha = results['mask'].squeeze()
        overlay[mask_alpha > 0.5] = [1, 0, 0]
        ax5.imshow(overlay)
        ax5.set_title('Robot Highlight\n(Red = YAM + Cube)', fontsize=11)
        ax5.axis('off')

        ax6 = plt.subplot(2, 4, 6)
        # Only robot and cube
        obj_view = results['foreground'].copy()
        obj_view[mask_alpha < 0.5] = 0
        ax6.imshow(obj_view)
        ax6.set_title('Objects Only\n(No Background)', fontsize=11)
        ax6.axis('off')

        ax7 = plt.subplot(2, 4, 7)
        ax7.imshow(results['mask'], cmap='hot')
        ax7.set_title('Blend Weights\n(Alpha Compositing)', fontsize=11)
        ax7.axis('off')

        # Statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        robot_ratio = results['mask'].mean()
        stats_text = f"""
MuGS + mjlab Demo

Task: YAM Lift Cube
Robot: i2rt YAM (6-DOF + Gripper)
Object: Red cube (2cm, 50g)

Resolution: {results['hybrid'].shape[1]}x{results['hybrid'].shape[0]}
Robot pixels: {robot_ratio*100:.2f}%
Background: {(1-robot_ratio)*100:.2f}%

Camera: robot/camera_d405
Mode: Hybrid rendering
Device: {device}

Episode: {env.episode_length_buf[0].item()}/{env.max_episode_length}
Reward: {reward[0].item():.3f}
"""
        ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('MuGS + mjlab: Photorealistic Background for Manipulation RL',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save
        output_path = output_dir / "yam_lift_comparison.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

        # Save individual images
        for name, img in results.items():
            img_path = output_dir / f"{name}.jpg"
            if img.ndim == 3:
                plt.imsave(img_path, img)
            else:
                plt.imsave(img_path, img.squeeze(), cmap='gray')
            print(f"✅ Saved: {img_path}")

        print("\n" + "=" * 80)
        print("✅ Demo complete!")
        print("=" * 80)
        print(f"\nOutputs saved to: {output_dir}")
        print("\nKey insight:")
        print("  MuJoCo alone: crude simulation background")
        print("  MuGS hybrid: photorealistic kitchen + physics simulation")
        print("  → Perfect for visual RL and VLA training!")

    else:
        print("❌ gaussian_rgb not found in observations")
        print(f"   Available observations: {list(obs_dict.keys())}")

    env.close()


if __name__ == "__main__":
    main()
