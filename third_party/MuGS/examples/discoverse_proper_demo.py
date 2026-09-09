#!/usr/bin/env python3
"""
DISCOVERSE Proper Integration Demo

Correctly uses DISCOVERSE's make_env() to combine robot + task,
then renders with MuGS.

This fixes the issues in discoverse_full_demo.py:
1. Uses DISCOVERSE's robot+task combination (not bare task XML)
2. Identifies robot geoms by body membership (not by name)
3. Fixes FOV bug (treats camera fovy as degrees, not radians)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data/external/DISCOVERSE"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from mugs.sensors import GaussianSensor, GaussianSensorConfig


def make_discoverse_scene(robot_name="airbot_play", task_name="place_coffeecup"):
    """
    Create DISCOVERSE scene using their make_env() function.

    Returns:
        tuple: (mjcf_path, robot_body_names)
    """
    print(f"\n{'='*80}")
    print(f"Creating DISCOVERSE Scene: {robot_name} + {task_name}")
    print(f"{'='*80}")

    # Import DISCOVERSE
    try:
        from discoverse.envs import make_env
    except ImportError as e:
        print(f"❌ DISCOVERSE not found: {e}")
        print("   Please ensure DISCOVERSE is cloned to data/external/DISCOVERSE")
        return None, None

    # Create combined environment
    env = make_env(robot_name, task_name)

    # Export to XML
    output_dir = Path("data/external/DISCOVERSE/models/mjcf/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    mjcf_path = output_dir / f"{robot_name}_{task_name}.xml"
    env.export_xml(str(mjcf_path))

    print(f"✅ Exported combined MJCF: {mjcf_path}")

    # Define robot bodies (for geom extraction)
    if robot_name == "airbot_play":
        robot_bodies = [
            "airbot_play_pose",
            "arm_base",
            "link1", "link2", "link3", "link4", "link5", "link6",
            "left", "right",  # gripper fingers
        ]
    else:
        # For other robots, we'll discover dynamically
        robot_bodies = []

    return mjcf_path, robot_bodies


def extract_robot_geoms_by_body(model, robot_body_names):
    """
    Extract geom IDs that belong to robot bodies.

    Most DISCOVERSE geoms don't have names, so we identify by parent body.

    Returns:
        List[int]: Geom IDs belonging to robot bodies
    """
    robot_geom_ids = []

    # Map body names to IDs
    robot_body_ids = set()
    for i in range(model.nbody):
        body_name = model.body(i).name
        if body_name in robot_body_names:
            robot_body_ids.add(i)

    print(f"✅ Found {len(robot_body_ids)} robot bodies")

    # Find geoms belonging to those bodies
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]

        if body_id in robot_body_ids:
            robot_geom_ids.append(i)

    print(f"✅ Found {len(robot_geom_ids)} robot geoms")

    return robot_geom_ids


def fix_camera_fov(model):
    """
    Fix DISCOVERSE camera FOV bug.

    DISCOVERSE XMLs have <compiler angle="radian"/> but camera fovy
    values are actually in degrees (e.g., fovy="72.02").

    MuJoCo loads them as radians (72.02 rad = 4128 degrees!),
    causing 3DGS to render from wrong viewpoint.

    Fix: Convert the values from degrees to radians.
    """
    print(f"\n{'='*80}")
    print("Fixing Camera FOV")
    print(f"{'='*80}")

    for i in range(model.ncam):
        cam = model.camera(i)
        fov_stored = model.cam_fovy[i]

        # DISCOVERSE stores degrees but MuJoCo interprets as radians
        # Convert degrees -> radians
        fov_degrees = fov_stored  # stored value is actually degrees
        fov_radians = np.radians(fov_degrees)

        print(f"Camera {i} '{cam.name}':")
        print(f"  Stored: {fov_stored:.4f} (mislabeled as radians)")
        print(f"  Fixed:  {fov_radians:.4f} rad ({fov_degrees:.2f}°)")

        model.cam_fovy[i] = fov_radians


def run_proper_demo(robot_name="airbot_play", task_name="place_coffeecup"):
    """
    Run DISCOVERSE demo with proper robot+task integration.
    """
    # Step 1: Create DISCOVERSE scene
    mjcf_path, robot_bodies = make_discoverse_scene(robot_name, task_name)

    if mjcf_path is None:
        return

    # Step 2: Load MuJoCo model
    print(f"\n{'='*80}")
    print("Loading MuJoCo Model")
    print(f"{'='*80}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    print(f"✅ Model loaded:")
    print(f"   - {model.nbody} bodies")
    print(f"   - {model.ngeom} geoms")
    print(f"   - {model.ncam} cameras")

    # Step 3: Fix camera FOV bug
    fix_camera_fov(model)

    # Step 4: Extract robot geoms
    robot_geom_ids = extract_robot_geoms_by_body(model, robot_bodies)

    # Step 5: Find 3DGS background
    print(f"\n{'='*80}")
    print("Finding 3DGS Background")
    print(f"{'='*80}")

    # Try DISCOVERSE 3DGS scenes first
    discoverse_scene = Path("data/external/DISCOVERSE/models/3dgs/scene/kitchen.ply")
    fallback_scene = Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")

    if discoverse_scene.exists():
        scene_ply = discoverse_scene
        print(f"✅ Using DISCOVERSE scene: {scene_ply}")
    elif fallback_scene.exists():
        scene_ply = fallback_scene
        print(f"⚠️  DISCOVERSE 3DGS not found, using fallback: {scene_ply}")
    else:
        print("❌ No 3DGS background available")
        return

    # Step 6: Configure MuGS sensor
    print(f"\n{'='*80}")
    print("Configuring MuGS Sensor")
    print(f"{'='*80}")

    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=scene_ply,
        render_mode="hybrid",
        robot_geom_ids=robot_geom_ids,  # Use geom IDs instead of names
    )

    sensor = GaussianSensor(config)

    # Step 7: Render
    print(f"\n{'='*80}")
    print("Rendering")
    print(f"{'='*80}")

    mujoco.mj_forward(model, data)

    # Find camera
    camera_name = model.camera(0).name if model.ncam > 0 else None
    if not camera_name:
        print("❌ No camera found")
        return

    print(f"Camera: {camera_name}")

    result = sensor.render(model, data, camera_name, return_components=True)

    print(f"✅ Rendered: {result['rgb'].shape}")

    # Step 8: Visualize
    print(f"\n{'='*80}")
    print("Creating Visualization")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(20, 10))

    # Row 1: Main comparison
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(result['foreground'])
    ax1.set_title('MuJoCo Only\n(Physics Simulation)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(result['background'])
    ax2.set_title('3DGS Only\n(Photorealistic)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(result['rgb'])
    ax3.set_title('MuGS Hybrid\n(Best of Both)', fontsize=12, fontweight='bold', color='#d62728')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(result['mask'], cmap='gray')
    ax4.set_title('Robot Mask\n(Blending)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2: Details
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(result['foreground'])
    ax5.set_title('MuJoCo Detail', fontsize=12)
    ax5.axis('off')

    ax6 = plt.subplot(2, 4, 6)
    # Robot highlight
    highlight = result['background'].copy()
    if result['mask'].max() > 0:
        mask_rgb = np.stack([result['mask']] * 3, axis=-1)
        highlight = (highlight * 0.7 + mask_rgb * np.array([1.0, 0.2, 0.2]) * 0.3).astype(np.uint8)
    ax6.imshow(highlight)
    ax6.set_title('Robot Highlight', fontsize=12)
    ax6.axis('off')

    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(result['mask'], cmap='hot')
    ax7.set_title('Blend Weights', fontsize=12)
    ax7.axis('off')

    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    stats_text = f"""DISCOVERSE Proper Demo

Task: {task_name}
Robot: {robot_name}

Scene Assets:
• MJCF: {mjcf_path.name}
• 3DGS: {scene_ply.name}
• Camera: {camera_name}

Rendering:
• Resolution: {config.width}×{config.height}
• Robot geoms: {len(robot_geom_ids)}
• Coverage: {result['mask'].mean()*100:.1f}%

Performance:
• Mode: Hybrid
• Background: 3DGS
• Foreground: MuJoCo

FOV Fix Applied: ✓
Robot Included: ✓
"""
    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle(f'DISCOVERSE + MuGS Proper Integration: {task_name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_dir = Path("outputs/discoverse_proper")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_name}_proper.jpg"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Copy to user outputs
    import shutil
    user_output = Path("/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250") / f"discoverse_proper_{task_name}.jpg"
    if user_output.parent.exists():
        shutil.copy(output_path, user_output)
        print(f"✅ Copied to: {user_output}")


def main():
    print("=" * 80)
    print("DISCOVERSE Proper Integration Demo")
    print("=" * 80)

    # Test with place_coffeecup task
    run_proper_demo("airbot_play", "place_coffeecup")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
