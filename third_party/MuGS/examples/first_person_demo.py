#!/usr/bin/env python3
"""
First-Person Gripper Demo with 3DGS Background

展示第一人称视角的机器人夹爪面对桌面抓取场景：
- 背景：kitchen.ply（照片级真实感）
- 前景：MuJoCo 物理物体 + 夹爪
- 视角：手腕第一人称

快速演示 MuGS 的核心价值：粗糙的 MuJoCo 背景 → 照片级 3DGS 背景
"""

import mujoco
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mugs.sensors import GaussianSensor, GaussianSensorConfig


def main():
    print("=" * 80)
    print("First-Person Gripper Demo")
    print("=" * 80)

    # Paths
    scene_path = Path(__file__).parent.parent / "scenes" / "first_person_kitchen.xml"
    ply_path = Path(__file__).parent.parent / "data" / "pretrained" / "kitchen" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    output_dir = Path(__file__).parent.parent / "outputs" / "first_person_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        print(f"❌ Background model not found: {ply_path}")
        print("   Run: python scripts/download_pretrained_models.py")
        return

    print(f"✅ Scene: {scene_path.name}")
    print(f"✅ Background: {ply_path}")

    # Load MuJoCo scene
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Foreground geoms (all MuJoCo objects to render: gripper + desktop items)
    foreground_geoms = [
        'gripper_base', 'gripper_left', 'gripper_right',  # Gripper
        'mug_body', 'mug_handle',  # Red mug
        'bowl_geom',  # Blue bowl
        'plate_geom',  # White plate
        'apple_geom',  # Green apple
    ]

    print(f"✅ Foreground geoms: {len(foreground_geoms)} objects")

    # Create sensor with 3DGS background
    print("\n" + "=" * 80)
    print("Creating GaussianSensor...")
    print("=" * 80)

    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=ply_path,
        render_mode="hybrid",
        robot_geom_names=foreground_geoms,  # All objects in foreground
    )
    sensor = GaussianSensor(config)

    print("✅ Sensor created")

    # Render different modes
    print("\n" + "=" * 80)
    print("Rendering comparisons...")
    print("=" * 80)

    results = {}

    # Load camera 117 parameters from kitchen.ply
    import json
    cam_params_path = Path(__file__).parent.parent / "data" / "pretrained" / "kitchen" / "cameras.json"
    with open(cam_params_path) as f:
        cameras = json.load(f)
    cam117 = cameras[117]  # Use camera 117 that looks at tabletop

    print(f"  Using Camera 117 from kitchen.ply training set")
    print(f"    Position: {cam117['position']}")

    # 1. Hybrid mode (our result)
    print("  [1/3] Hybrid (3DGS background + MuJoCo foreground)...")
    sensor.cfg.render_mode = "hybrid"
    result_hybrid = sensor.render(model, data, "wrist_cam", return_components=True, camera_params=cam117)

    # Add seg_ids if not present
    if 'seg_ids' not in result_hybrid:
        # Re-render MuJoCo to get segmentation
        foreground_rgb, seg_ids = sensor._render_mujoco_foreground(model, data, "wrist_cam")
        result_hybrid['seg_ids'] = seg_ids[:, :, np.newaxis]

    results['hybrid'] = result_hybrid

    # 2. MuJoCo only (baseline) - use same camera params
    print("  [2/3] MuJoCo only (baseline)...")
    sensor.cfg.render_mode = "mujoco_only"
    result_mujoco = sensor.render(model, data, "wrist_cam", camera_params=cam117)
    results['mujoco'] = {'rgb': result_mujoco}  # Wrap as dict

    # 3. 3DGS only (background) - use same camera params
    print("  [3/3] 3DGS only (background)...")
    sensor.cfg.render_mode = "3dgs_only"
    result_3dgs = sensor.render(model, data, "wrist_cam", camera_params=cam117)
    results['3dgs'] = {'rgb': result_3dgs}  # Wrap as dict

    # Visualization
    print("\n" + "=" * 80)
    print("Generating visualization...")
    print("=" * 80)

    # Extract commonly used data
    mask = results['hybrid']['mask']
    seg_vis = results['hybrid']['seg_ids']

    fig = plt.figure(figsize=(24, 16))

    # Row 1: Main Comparison
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(results['mujoco']['rgb'])
    ax1.set_title('MuJoCo Only\nCrude Simulation Background', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(results['3dgs']['rgb'])
    ax2.set_title('3DGS Only\nPhotorealistic Background', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(results['hybrid']['rgb'])
    ax3.set_title('MuGS Hybrid (Our Result)\nRealistic + Physics', fontsize=12, fontweight='bold', color='#d62728')
    ax3.axis('off')

    # Segment ID visualization
    ax4 = plt.subplot(3, 4, 4)
    seg_vis = results['hybrid']['seg_ids']
    ax4.imshow(seg_vis, cmap='tab20')
    ax4.set_title('Segment IDs\n(Object Identification)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2: Components
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(results['hybrid']['background'])
    ax5.set_title('3DGS Background\n(Real Kitchen Scene)', fontsize=11)
    ax5.axis('off')

    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(results['hybrid']['foreground'])
    ax6.set_title('MuJoCo Foreground\n(Simulated Objects)', fontsize=11)
    ax6.axis('off')

    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(results['hybrid']['mask'], cmap='gray')
    ax7.set_title('Robot Mask\n(Gripper Region)', fontsize=11)
    ax7.axis('off')

    # Segment ID with labels
    ax8 = plt.subplot(3, 4, 8)
    seg_colored = plt.cm.tab20(seg_vis.squeeze() / 20.0)[:,:,:3]
    ax8.imshow(seg_colored)
    ax8.set_title('Colored Segments\n(Per-Object Mask)', fontsize=11)
    ax8.axis('off')

    # Row 3: Detailed views
    ax9 = plt.subplot(3, 4, 9)
    # Show only objects (no floor)
    obj_mask = (seg_vis > 0) & (seg_vis < 100)
    obj_view = results['hybrid']['foreground'].copy()
    obj_view[~obj_mask.squeeze()] = 0
    ax9.imshow(obj_view)
    ax9.set_title('Objects Only\n(No Background)', fontsize=11)
    ax9.axis('off')

    ax10 = plt.subplot(3, 4, 10)
    # Overlay mask on background
    overlay = results['hybrid']['background'].copy().astype(float) / 255
    mask_alpha = results['hybrid']['mask'].squeeze()
    overlay[mask_alpha > 0.5] = [1, 0, 0]  # Red highlight for robot
    ax10.imshow(overlay)
    ax10.set_title('Robot Highlight\n(Red = Gripper)', fontsize=11)
    ax10.axis('off')

    ax11 = plt.subplot(3, 4, 11)
    # Show blend weights
    blend = results['hybrid']['mask']
    ax11.imshow(blend, cmap='hot')
    ax11.set_title('Blend Weights\n(Alpha Compositing)', fontsize=11)
    ax11.axis('off')

    # Statistics text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    stats_text = f"""
Demo Statistics:

Resolution: {results['hybrid']['rgb'].shape[1]}x{results['hybrid']['rgb'].shape[0]}
Robot pixels: {mask.mean()*100:.2f}%
Background: {(1-mask.mean())*100:.2f}%

Unique segments: {len(np.unique(seg_vis))}

Camera 117 params:
  Position: [1.56, 1.21, -2.34]
  FOV: 35.55°
  Aspect: 1.499

Objects visible:
  • Mug (red cylinder)
  • Bowl (blue cylinder)
  • Plate (white disk)
  • Apple (green sphere)
  • Gripper (gray boxes)
"""
    ax12.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('MuGS First-Person Demo: Photorealistic Background + Physics Simulation',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    output_path = output_dir / "first_person_comparison.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Save individual images
    for name, result in results.items():
        img_path = output_dir / f"{name}.jpg"
        plt.imsave(img_path, result['rgb'])
        print(f"✅ Saved: {img_path}")

    # Stats
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)

    mask = results['hybrid']['mask']
    robot_ratio = mask.mean()
    print(f"  Robot pixels: {robot_ratio*100:.2f}%")
    print(f"  Background pixels: {(1-robot_ratio)*100:.2f}%")
    print(f"  Image size: {results['hybrid']['rgb'].shape}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nKey insight:")
    print("  左图（MuJoCo）: 粗糙的仿真背景")
    print("  右图（MuGS）:  照片级真实背景 + 同样的物理物体")
    print("  → 完美的 Sim2Real VLA 训练场景！")


if __name__ == "__main__":
    main()
