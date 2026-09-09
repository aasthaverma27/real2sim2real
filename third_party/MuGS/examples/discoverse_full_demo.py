#!/usr/bin/env python3
"""
Complete DISCOVERSE Assets Demo with MuGS

This demo shows:
1. How to clone and set up DISCOVERSE
2. Load DISCOVERSE task environments
3. Use DISCOVERSE 3DGS backgrounds
4. Render with MuGS hybrid mode
5. Generate comparison visualizations

Author: MuGS Team
"""

import sys
import os
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

from mugs.sensors import GaussianSensor, GaussianSensorConfig


# Configuration
DISCOVERSE_DIR = Path("data/external/DISCOVERSE")
OUTPUT_DIR = Path("outputs/discoverse_demo")


class DISCOVERSESetup:
    """Helper class for DISCOVERSE setup and asset management."""

    @staticmethod
    def clone_repository():
        """Clone DISCOVERSE repository if not exists."""
        if DISCOVERSE_DIR.exists():
            print(f"✅ DISCOVERSE already exists at {DISCOVERSE_DIR}")
            return True

        print("=" * 80)
        print("Cloning DISCOVERSE Repository")
        print("=" * 80)

        DISCOVERSE_DIR.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"📥 Cloning to {DISCOVERSE_DIR}...")
            subprocess.run(
                ["git", "clone", "https://github.com/TATP-233/DISCOVERSE", str(DISCOVERSE_DIR)],
                check=True,
            )

            # Try to pull LFS if available
            print("\n📥 Attempting to pull LFS assets...")
            subprocess.run(
                ["git", "lfs", "pull"],
                cwd=str(DISCOVERSE_DIR),
                check=False,  # Don't fail if LFS not installed
            )

            print("✅ DISCOVERSE cloned successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone DISCOVERSE: {e}")
            print("\nPlease clone manually:")
            print(f"  git clone https://github.com/TATP-233/DISCOVERSE {DISCOVERSE_DIR}")
            return False

    @staticmethod
    def check_setup():
        """Check if DISCOVERSE is properly set up."""
        if not DISCOVERSE_DIR.exists():
            return False

        # Check for key directories
        required_dirs = [
            DISCOVERSE_DIR / "models",
            DISCOVERSE_DIR / "models/mjcf",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"⚠️  Missing directory: {dir_path}")
                return False

        return True


class DISCOVERSEAssetManager:
    """Manage DISCOVERSE assets (MJCF, 3DGS, etc.)."""

    def __init__(self, discoverse_dir: Path = DISCOVERSE_DIR):
        self.discoverse_dir = discoverse_dir
        self.mjcf_dir = discoverse_dir / "models/mjcf"
        self.gs_dir = discoverse_dir / "models/3dgs"

    def list_tasks(self) -> List[Dict]:
        """List all available task MJCF files."""
        tasks = []

        task_dirs = [
            ("task_environments", self.mjcf_dir / "task_environments"),
            ("tasks_mmk2", self.mjcf_dir / "tasks_mmk2"),
            ("tasks_airbot_play", self.mjcf_dir / "tasks_airbot_play"),
        ]

        for category, task_dir in task_dirs:
            if not task_dir.exists():
                continue

            for xml_file in task_dir.glob("*.xml"):
                tasks.append({
                    'name': xml_file.stem,
                    'path': xml_file,
                    'category': category,
                })

        return tasks

    def find_3dgs_scenes(self) -> List[Path]:
        """Find all available 3DGS scene files."""
        scene_dir = self.gs_dir / "scene"

        if not scene_dir.exists():
            print(f"⚠️  3DGS scene directory not found: {scene_dir}")
            return []

        return list(scene_dir.glob("*.ply"))

    def get_fallback_3dgs(self) -> Optional[Path]:
        """Get fallback 3DGS (pretrained kitchen)."""
        fallback = Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")
        return fallback if fallback.exists() else None

    def extract_robot_geoms(self, mjcf_path: Path) -> List[str]:
        """Extract robot geom names from MJCF."""
        try:
            # Try XML parsing first
            tree = ET.parse(mjcf_path)
            root = tree.getroot()

            geoms = []
            for geom in root.iter('geom'):
                name = geom.get('name')
                # Filter out floor/ground/table
                if name and not any(kw in name.lower() for kw in ['floor', 'ground', 'table', 'wall']):
                    geoms.append(name)

            if geoms:
                return geoms

        except Exception as e:
            print(f"⚠️  XML parsing failed: {e}, trying MuJoCo...")

        # Fallback: load with MuJoCo
        try:
            model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            geoms = []
            for i in range(model.ngeom):
                name = model.geom(i).name
                if name and not any(kw in name.lower() for kw in ['floor', 'ground', 'table', 'wall']):
                    geoms.append(name)
            return geoms

        except Exception as e:
            print(f"❌ Failed to extract geoms: {e}")
            return []


def run_demo_task(
    task_path: Path,
    scene_ply: Optional[Path] = None,
    asset_manager: Optional[DISCOVERSEAssetManager] = None,
):
    """
    Run a single DISCOVERSE task with MuGS rendering.

    Args:
        task_path: Path to task MJCF file
        scene_ply: Path to 3DGS scene (optional)
        asset_manager: Asset manager instance
    """
    print("\n" + "=" * 80)
    print(f"Demo: {task_path.stem}")
    print("=" * 80)

    if asset_manager is None:
        asset_manager = DISCOVERSEAssetManager()

    # Load MuJoCo model
    print(f"\n📋 Loading MJCF: {task_path.name}")
    try:
        model = mujoco.MjModel.from_xml_path(str(task_path))
        data = mujoco.MjData(model)
        print(f"   ✓ Model loaded: {model.nbody} bodies, {model.ngeom} geoms")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return

    # Find cameras
    cameras = []
    for i in range(model.ncam):
        cam_name = model.camera(i).name
        if cam_name:
            cameras.append(cam_name)

    if not cameras:
        print("   ⚠️  No cameras in scene, cannot render")
        return

    print(f"   ✓ Found {len(cameras)} camera(s): {cameras}")
    camera_name = cameras[0]

    # Extract robot geoms
    print(f"\n🤖 Extracting robot geoms...")
    robot_geoms = asset_manager.extract_robot_geoms(task_path)
    print(f"   ✓ Found {len(robot_geoms)} geoms")
    if robot_geoms:
        print(f"   Examples: {robot_geoms[:5]}")

    # Find 3DGS background
    if scene_ply is None:
        print(f"\n🎨 Finding 3DGS background...")
        scenes = asset_manager.find_3dgs_scenes()
        if scenes:
            scene_ply = scenes[0]
            print(f"   ✓ Using: {scene_ply.name}")
        else:
            fallback = asset_manager.get_fallback_3dgs()
            if fallback:
                scene_ply = fallback
                print(f"   ⚠️  Using fallback: {fallback}")
            else:
                print(f"   ❌ No 3DGS background available")
                return
    else:
        print(f"\n🎨 Using provided 3DGS: {scene_ply.name}")

    # Create MuGS sensor
    print(f"\n🖼️  Setting up MuGS sensor...")
    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=scene_ply,
        render_mode="hybrid",
        robot_geom_names=robot_geoms,
    )

    sensor = GaussianSensor(config)
    print(f"   ✓ Sensor initialized")

    # Run simulation and render
    print(f"\n⚙️  Running simulation...")
    mujoco.mj_forward(model, data)

    print(f"📸 Rendering from camera: {camera_name}")
    result = sensor.render(model, data, camera_name, return_components=True)
    print(f"   ✓ Rendered: {result['rgb'].shape}")

    # Calculate statistics
    robot_coverage = result['mask'].mean() * 100
    print(f"\n📊 Statistics:")
    print(f"   Robot coverage: {robot_coverage:.2f}%")
    print(f"   Background coverage: {100-robot_coverage:.2f}%")

    # Visualize
    print(f"\n🎨 Creating visualization...")
    fig = plt.figure(figsize=(20, 10))

    # Row 1: Main comparison
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(result['foreground'])
    ax1.set_title('MuJoCo Only\n(Physics Simulation)', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(result['background'])
    ax2.set_title('3DGS Only\n(Photorealistic)', fontsize=13, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(result['rgb'])
    ax3.set_title('MuGS Hybrid\n(Best of Both)', fontsize=13, fontweight='bold', color='#d62728')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(result['mask'], cmap='gray')
    ax4.set_title('Robot Mask\n(Blending)', fontsize=13)
    ax4.axis('off')

    # Row 2: Details
    ax5 = plt.subplot(2, 4, 5)
    # Crop robot detail
    h, w = result['foreground'].shape[:2]
    if robot_coverage > 1:  # Only if robot is visible
        mask = result['mask'].squeeze()
        ys, xs = np.where(mask > 0.5)
        if len(ys) > 0:
            y1, y2 = max(0, ys.min()-20), min(h, ys.max()+20)
            x1, x2 = max(0, xs.min()-20), min(w, xs.max()+20)
            crop = result['foreground'][y1:y2, x1:x2]
            ax5.imshow(crop)
        else:
            ax5.imshow(result['foreground'])
    else:
        ax5.imshow(result['foreground'])
    ax5.set_title('MuJoCo Detail', fontsize=11)
    ax5.axis('off')

    ax6 = plt.subplot(2, 4, 6)
    # Overlay robot on background
    overlay = result['background'].copy().astype(float) / 255
    mask_alpha = result['mask'].squeeze()
    overlay[mask_alpha > 0.5] = [1, 0.3, 0.3]  # Red highlight
    ax6.imshow(overlay)
    ax6.set_title('Robot Highlight', fontsize=11)
    ax6.axis('off')

    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(result['mask'], cmap='hot')
    ax7.set_title('Blend Weights', fontsize=11)
    ax7.axis('off')

    # Statistics panel
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    stats_text = f"""
DISCOVERSE Task Demo

Task: {task_path.stem}
Category: {task_path.parent.name}

Scene Assets:
• MJCF: {task_path.name}
• 3DGS: {scene_ply.name}
• Camera: {camera_name}

Rendering:
• Resolution: {w}×{h}
• Robot geoms: {len(robot_geoms)}
• Coverage: {robot_coverage:.1f}%

Performance:
• Mode: Hybrid
• Background: 3DGS
• Foreground: MuJoCo
"""
    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle(f'DISCOVERSE + MuGS Demo: {task_path.stem}',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = OUTPUT_DIR / f"{task_path.stem}_demo.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

    # Copy to user outputs
    user_output_dir = Path("/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250")
    if user_output_dir.exists():
        import shutil
        user_output = user_output_dir / f"discoverse_{task_path.stem}.jpg"
        shutil.copy(output_path, user_output)
        print(f"   ✓ Copied to: {user_output.name}")

    print(f"\n✅ Demo complete for {task_path.stem}")


def main():
    """Main demo entry point."""
    print("=" * 80)
    print("DISCOVERSE Assets Demo with MuGS")
    print("Complete demonstration of DISCOVERSE integration")
    print("=" * 80)

    # Step 1: Setup DISCOVERSE
    print("\n📦 Step 1: Setup DISCOVERSE")
    print("-" * 80)

    if not DISCOVERSESetup.check_setup():
        print("DISCOVERSE not found, attempting to clone...")
        if not DISCOVERSESetup.clone_repository():
            print("\n❌ Setup failed. Please install DISCOVERSE manually.")
            return

    if not DISCOVERSESetup.check_setup():
        print("\n❌ DISCOVERSE setup incomplete")
        return

    print("✅ DISCOVERSE is set up")

    # Step 2: List available assets
    print("\n📋 Step 2: Discover Available Assets")
    print("-" * 80)

    asset_manager = DISCOVERSEAssetManager()

    # List tasks
    tasks = asset_manager.list_tasks()
    print(f"\n✅ Found {len(tasks)} tasks:")
    for i, task in enumerate(tasks[:10], 1):  # Show first 10
        print(f"   {i}. {task['name']:<30} ({task['category']})")
    if len(tasks) > 10:
        print(f"   ... and {len(tasks)-10} more")

    # List 3DGS scenes
    scenes = asset_manager.find_3dgs_scenes()
    print(f"\n✅ Found {len(scenes)} 3DGS scenes:")
    for scene in scenes[:5]:  # Show first 5
        size_mb = scene.stat().st_size / 1024 / 1024
        print(f"   • {scene.name} ({size_mb:.1f} MB)")

    if not scenes:
        print("   ⚠️  No 3DGS scenes found (will auto-download on first DISCOVERSE run)")
        fallback = asset_manager.get_fallback_3dgs()
        if fallback:
            print(f"   ✓ Using fallback: {fallback}")
            scenes = [fallback]
        else:
            print("   ❌ No 3DGS available")

    # Step 3: Run demos
    print("\n🚀 Step 3: Run Task Demos")
    print("-" * 80)

    if not tasks:
        print("❌ No tasks available to demo")
        return

    if not scenes:
        print("❌ No 3DGS backgrounds available")
        return

    # Demo first 3 tasks or all available
    demo_count = min(3, len(tasks))
    print(f"\nRunning {demo_count} task demo(s)...")

    for i, task in enumerate(tasks[:demo_count], 1):
        print(f"\n[{i}/{demo_count}] Processing: {task['name']}")
        run_demo_task(
            task_path=task['path'],
            scene_ply=scenes[0] if scenes else None,
            asset_manager=asset_manager,
        )

    # Summary
    print("\n" + "=" * 80)
    print("✅ DISCOVERSE Demo Complete!")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"   • Tasks demonstrated: {demo_count}")
    print(f"   • Total tasks available: {len(tasks)}")
    print(f"   • 3DGS scenes available: {len(scenes)}")
    print(f"   • Output directory: {OUTPUT_DIR}")

    print(f"\n📁 Generated files:")
    for output_file in sorted(OUTPUT_DIR.glob("*.jpg")):
        print(f"   • {output_file.name}")

    print(f"\n💡 Next steps:")
    print(f"   1. View outputs in: {OUTPUT_DIR}")
    print(f"   2. Try other tasks by modifying this script")
    print(f"   3. Read docs: docs/DISCOVERSE_INTEGRATION.md")
    print(f"   4. Create custom scenes with DISCOVERSE backgrounds")


if __name__ == "__main__":
    main()
