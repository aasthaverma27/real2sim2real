#!/usr/bin/env python3
"""
DISCOVERSE Task Integration Demo

Demonstrates loading DISCOVERSE manipulation tasks and rendering with MuGS.

Prerequisites:
    git clone https://github.com/TATP-233/DISCOVERSE data/external/DISCOVERSE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from mugs.sensors import GaussianSensor, GaussianSensorConfig


DISCOVERSE_DIR = Path("data/external/DISCOVERSE")


def check_discoverse_installed():
    """Check if DISCOVERSE is installed."""
    if not DISCOVERSE_DIR.exists():
        print("=" * 80)
        print("❌ DISCOVERSE not found")
        print("=" * 80)
        print("\nPlease install DISCOVERSE:")
        print(f"\n  git clone https://github.com/TATP-233/DISCOVERSE {DISCOVERSE_DIR}")
        print("  cd data/external/DISCOVERSE")
        print("  git lfs install")
        print("  git lfs pull")
        print("\n" + "=" * 80)
        return False
    return True


def list_available_tasks():
    """List all available DISCOVERSE tasks."""
    print("\n" + "=" * 80)
    print("Available DISCOVERSE Tasks")
    print("=" * 80)

    task_dirs = [
        DISCOVERSE_DIR / "models/mjcf/task_environments",
        DISCOVERSE_DIR / "models/mjcf/tasks_mmk2",
    ]

    tasks = []
    for task_dir in task_dirs:
        if task_dir.exists():
            for xml_file in task_dir.glob("*.xml"):
                tasks.append({
                    'name': xml_file.stem,
                    'path': xml_file,
                    'category': task_dir.name,
                })

    if not tasks:
        print("⚠️  No task XML files found")
        print(f"   Expected in: {task_dirs[0]}")
        return []

    for task in tasks:
        print(f"  - {task['name']:<25} ({task['category']})")

    return tasks


def extract_geom_names(mjcf_path):
    """Extract geom names from MJCF file."""
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        geoms = []
        for geom in root.iter('geom'):
            name = geom.get('name')
            if name and not name.startswith('floor'):
                geoms.append(name)

        return geoms
    except Exception as e:
        print(f"⚠️  Could not parse MJCF: {e}")
        # Fallback: load with MuJoCo and extract
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        geoms = []
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name and not name.startswith('floor'):
                geoms.append(name)
        return geoms


def find_3dgs_scene():
    """Find available 3DGS scene files."""
    scene_dir = DISCOVERSE_DIR / "models/3dgs/scene"

    if not scene_dir.exists():
        print(f"⚠️  3DGS scene directory not found: {scene_dir}")
        print("   3DGS files will be auto-downloaded on first DISCOVERSE run")
        return None

    ply_files = list(scene_dir.glob("*.ply"))

    if not ply_files:
        print(f"⚠️  No PLY files in {scene_dir}")
        return None

    print(f"✅ Found {len(ply_files)} 3DGS scenes")
    return ply_files[0]


def run_task_demo(task_xml_path, scene_ply_path=None):
    """
    Run a DISCOVERSE task with MuGS rendering.

    Args:
        task_xml_path: Path to task MJCF file
        scene_ply_path: Path to 3DGS background (optional)
    """
    print("\n" + "=" * 80)
    print(f"Running Task: {task_xml_path.stem}")
    print("=" * 80)

    # Load MuJoCo model
    print(f"Loading MJCF: {task_xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(task_xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ Failed to load MJCF: {e}")
        return

    print(f"✅ Loaded model: {model.ngeom} geoms, {model.nbody} bodies")

    # List cameras
    camera_names = []
    for i in range(model.ncam):
        cam_name = model.camera(i).name
        if cam_name:
            camera_names.append(cam_name)

    if not camera_names:
        print("⚠️  No cameras found in scene, using default view")
        # Add a simple camera
        print("   You may want to add a camera to the MJCF file")
        return

    print(f"✅ Cameras: {camera_names}")
    camera_name = camera_names[0]

    # Extract geom names
    geom_names = extract_geom_names(task_xml_path)
    print(f"✅ Found {len(geom_names)} geoms for foreground")

    # Find or use provided 3DGS scene
    if scene_ply_path is None:
        scene_ply_path = find_3dgs_scene()

    if scene_ply_path is None:
        # Fallback to pretrained kitchen
        fallback = Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")
        if fallback.exists():
            print(f"⚠️  Using fallback 3DGS: {fallback}")
            scene_ply_path = fallback
        else:
            print("❌ No 3DGS background available")
            return

    print(f"✅ 3DGS background: {scene_ply_path}")

    # Configure MuGS
    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=scene_ply_path,
        render_mode="hybrid",
        robot_geom_names=geom_names,
    )

    sensor = GaussianSensor(config)

    # Forward simulation
    mujoco.mj_forward(model, data)

    # Render
    print(f"Rendering from camera: {camera_name}")
    result = sensor.render(model, data, camera_name, return_components=True)

    print(f"✅ Rendered: {result['rgb'].shape}")

    # Visualize
    fig = plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(result['foreground'])
    ax1.set_title('MuJoCo Only\n(DISCOVERSE Task)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(result['background'])
    ax2.set_title('3DGS Background\n(DISCOVERSE Scene)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(result['rgb'])
    ax3.set_title('MuGS Hybrid\n(DISCOVERSE + MuGS)', fontsize=14, fontweight='bold', color='#d62728')
    ax3.axis('off')

    plt.suptitle(f'DISCOVERSE Task: {task_xml_path.stem}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_dir = Path("outputs/discoverse_tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_xml_path.stem}.jpg"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Also copy to user outputs
    import shutil
    user_output = Path("/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250") / f"discoverse_{task_xml_path.stem}.jpg"
    if user_output.parent.exists():
        shutil.copy(output_path, user_output)
        print(f"✅ Copied to: {user_output}")


def main():
    print("=" * 80)
    print("DISCOVERSE Task Demo with MuGS")
    print("=" * 80)

    # Check installation
    if not check_discoverse_installed():
        return

    # List available tasks
    tasks = list_available_tasks()

    if not tasks:
        print("\n⚠️  No tasks found. DISCOVERSE may need to be set up.")
        print("    Try running DISCOVERSE examples first to download assets.")
        return

    # Run first available task as demo
    if tasks:
        print(f"\n📋 Running demo with: {tasks[0]['name']}")
        run_task_demo(tasks[0]['path'])

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)

    print("\nTo run specific tasks:")
    for task in tasks[:5]:  # Show first 5
        print(f"  - {task['name']}")

    print("\nModify this script to run other tasks:")
    print("  run_task_demo(Path('data/external/DISCOVERSE/models/mjcf/tasks_mmk2/pan_pick.xml'))")


if __name__ == "__main__":
    main()
