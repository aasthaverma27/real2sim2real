#!/usr/bin/env python3
"""
Example: Using External 3DGS Assets with MuGS

Demonstrates how to load and use assets from:
- GS-Playground
- DISCOVERSE
- Bridge-GS dataset
- Custom 3DGS scenes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mugs.sensors import GaussianSensor, GaussianSensorConfig


def example1_bridge_gs_scene():
    """
    Example 1: Load a Bridge-GS manipulation scene

    Prerequisites:
        python scripts/download_external_assets.py bridge-gs
    """
    print("=" * 80)
    print("Example 1: Bridge-GS Scene")
    print("=" * 80)

    # Path to Bridge-GS asset (update based on actual structure)
    bridge_gs_dir = Path("data/external/bridge-gs")

    # Check if assets exist
    if not bridge_gs_dir.exists():
        print(f"❌ Bridge-GS assets not found at {bridge_gs_dir}")
        print("   Download with: python scripts/download_external_assets.py bridge-gs")
        return

    # For this example, we'll use a simple robot with external background
    # In practice, you'd load the full Bridge-GS task specification

    robot_xml = """
    <mujoco model="bridge_robot">
      <compiler angle="radian"/>
      <visual>
        <headlight ambient="0.5 0.5 0.5"/>
      </visual>
      <worldbody>
        <light pos="0.5 0 1.5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <camera name="main_cam" pos="0.6 -0.5 0.5" xyaxes="0.8 0.6 0 -0.4 0.5 0.8" fovy="0.7854"/>

        <!-- Simple gripper -->
        <body name="gripper" pos="0.3 0 0.2">
          <geom name="palm" type="box" size="0.03 0.04 0.01" rgba="0.7 0.7 0.7 1"/>
          <body name="left_finger" pos="-0.04 0 0">
            <geom name="left_finger_geom" type="box" size="0.005 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
          </body>
          <body name="right_finger" pos="0.04 0 0">
            <geom name="right_finger_geom" type="box" size="0.005 0.02 0.03" rgba="0.6 0.6 0.6 1"/>
          </body>
        </body>

        <!-- Object to manipulate -->
        <body name="cube" pos="0.35 0.1 0.11">
          <joint type="free"/>
          <geom name="cube_geom" type="box" size="0.02 0.02 0.02" rgba="0.9 0.2 0.1 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(robot_xml)
        scene_path = f.name

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Try to find a PLY file in Bridge-GS
    ply_files = list(bridge_gs_dir.rglob("*.ply"))
    if ply_files:
        ply_path = ply_files[0]
        print(f"✅ Found 3DGS asset: {ply_path}")
    else:
        print(f"⚠️  No PLY files found in {bridge_gs_dir}")
        print("   Using fallback (pretrained kitchen)")
        ply_path = Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")

    if not ply_path.exists():
        print(f"❌ 3DGS asset not found: {ply_path}")
        return

    # Configure sensor
    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=ply_path,
        render_mode="hybrid",
        robot_geom_names=["palm", "left_finger_geom", "right_finger_geom", "cube_geom"],
    )

    sensor = GaussianSensor(config)
    result = sensor.render(model, data, "main_cam", return_components=True)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(result['foreground'])
    axes[0].set_title('MuJoCo Only')
    axes[0].axis('off')

    axes[1].imshow(result['background'])
    axes[1].set_title('3DGS Background (Bridge-GS)')
    axes[1].axis('off')

    axes[2].imshow(result['rgb'])
    axes[2].set_title('MuGS Hybrid')
    axes[2].axis('off')

    plt.tight_layout()
    output_path = Path("outputs/external_assets_demo/bridge_gs_example.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def example2_discoverse_integration():
    """
    Example 2: Integration with DISCOVERSE scenes

    Prerequisites:
        git clone https://github.com/TATP-233/DISCOVERSE data/external/DISCOVERSE
    """
    print("\n" + "=" * 80)
    print("Example 2: DISCOVERSE Integration")
    print("=" * 80)

    discoverse_dir = Path("data/external/DISCOVERSE")

    if not discoverse_dir.exists():
        print(f"❌ DISCOVERSE not found at {discoverse_dir}")
        print("   Download with: python scripts/download_external_assets.py discoverse")
        print("   Or clone: git clone https://github.com/TATP-233/DISCOVERSE data/external/DISCOVERSE")
        return

    print(f"✅ DISCOVERSE directory found")
    print(f"   Looking for scene files...")

    # Look for MJCF scene files
    mjcf_files = list(discoverse_dir.rglob("*.xml"))
    if mjcf_files:
        print(f"   Found {len(mjcf_files)} MJCF files")
        print(f"   Example: {mjcf_files[0]}")

    # Look for PLY files
    ply_files = list(discoverse_dir.rglob("*.ply"))
    if ply_files:
        print(f"   Found {len(ply_files)} PLY files")
        print(f"   Example: {ply_files[0]}")

    print("\n💡 To use DISCOVERSE scenes:")
    print("   1. Load the MJCF scene file with mujoco.MjModel.from_xml_path()")
    print("   2. Extract robot geom names from the scene")
    print("   3. Use the corresponding PLY file as background")
    print("   4. Render with MuGS")


def example3_custom_scene_with_external_bg():
    """
    Example 3: Custom robot with external 3DGS background

    Shows how to combine your own robot MJCF with downloaded backgrounds.
    """
    print("\n" + "=" * 80)
    print("Example 3: Custom Scene + External Background")
    print("=" * 80)

    # Try different background sources in priority order
    background_sources = [
        ("Bridge-GS", "data/external/bridge-gs/**/*.ply"),
        ("InteriorGS", "data/external/interior-gs/**/*.ply"),
        ("DISCOVERSE", "data/external/DISCOVERSE/**/*.ply"),
        ("Pretrained", "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"),
    ]

    ply_path = None
    source_name = None

    for name, pattern in background_sources:
        paths = list(Path(".").glob(pattern))
        if paths:
            ply_path = paths[0]
            source_name = name
            break

    if not ply_path or not ply_path.exists():
        print("❌ No 3DGS backgrounds found")
        print("   Download with: python scripts/download_external_assets.py <source>")
        return

    print(f"✅ Using background from: {source_name}")
    print(f"   Path: {ply_path}")

    # Your custom robot
    robot_xml = """
    <mujoco model="custom_robot">
      <compiler angle="radian"/>
      <visual><headlight ambient="0.5 0.5 0.5"/></visual>
      <worldbody>
        <light pos="0.5 0 1.5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <camera name="view" pos="0.6 -0.6 0.6" xyaxes="0.7 0.7 0 -0.4 0.4 0.8" fovy="0.7854"/>

        <!-- Simple manipulator -->
        <body name="base" pos="0 0 0.1">
          <geom name="base_geom" type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.3 1"/>
          <body name="arm" pos="0 0 0.1">
            <joint name="shoulder" type="hinge" axis="0 0 1"/>
            <geom name="arm_geom" type="capsule" size="0.03" fromto="0 0 0 0.2 0 0" rgba="0.6 0.6 0.6 1"/>
            <body name="forearm" pos="0.2 0 0">
              <joint name="elbow" type="hinge" axis="0 1 0"/>
              <geom name="forearm_geom" type="capsule" size="0.025" fromto="0 0 0 0.15 0 0" rgba="0.5 0.5 0.5 1"/>
            </body>
          </body>
        </body>

        <body name="target" pos="0.3 0.1 0.15">
          <geom name="target_geom" type="sphere" size="0.03" rgba="1 0 0 1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="m1" joint="shoulder" gear="10"/>
        <motor name="m2" joint="elbow" gear="10"/>
      </actuator>
    </mujoco>
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(robot_xml)
        scene_path = f.name

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Set pose
    data.qpos[0] = 0.5  # shoulder
    data.qpos[1] = -0.3  # elbow
    mujoco.mj_forward(model, data)

    # Render with external background
    config = GaussianSensorConfig(
        width=640,
        height=480,
        background_ply_path=ply_path,
        render_mode="hybrid",
        robot_geom_names=["base_geom", "arm_geom", "forearm_geom", "target_geom"],
    )

    sensor = GaussianSensor(config)
    result = sensor.render(model, data, "view", return_components=True)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(result['foreground'])
    axes[0, 0].set_title(f'Custom Robot (MuJoCo)', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['background'])
    axes[0, 1].set_title(f'Background ({source_name})', fontsize=14)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(result['rgb'])
    axes[1, 0].set_title('MuGS Hybrid', fontsize=14, fontweight='bold', color='#d62728')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['mask'], cmap='gray')
    axes[1, 1].set_title('Robot Mask', fontsize=14)
    axes[1, 1].axis('off')

    plt.suptitle(f'Custom Robot + {source_name} Background', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = Path("outputs/external_assets_demo/custom_scene_example.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def example4_performance_comparison():
    """
    Example 4: Performance comparison between different asset sources
    """
    print("\n" + "=" * 80)
    print("Example 4: Performance Comparison")
    print("=" * 80)

    # Find all available PLY files
    sources = {
        "Pretrained Kitchen": "data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply",
        "Bridge-GS": "data/external/bridge-gs/**/*.ply",
        "InteriorGS": "data/external/interior-gs/**/*.ply",
    }

    available = {}
    for name, pattern in sources.items():
        if "**" in pattern:
            paths = list(Path(".").glob(pattern))
            if paths:
                available[name] = paths[0]
        else:
            path = Path(pattern)
            if path.exists():
                available[name] = path

    if not available:
        print("❌ No 3DGS assets found for comparison")
        return

    print(f"✅ Found {len(available)} asset sources:")
    for name, path in available.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"   - {name}: {size_mb:.1f} MB")

    print("\n💡 Performance comparison requires rendering benchmarks")
    print("   Run with actual rendering to measure FPS for each source")


def main():
    print("MuGS External Assets Examples")
    print("=" * 80)

    # Run examples
    example1_bridge_gs_scene()
    example2_discoverse_integration()
    example3_custom_scene_with_external_bg()
    example4_performance_comparison()

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Download more assets: python scripts/download_external_assets.py list")
    print("  2. Read docs: docs/EXTERNAL_ASSETS.md")
    print("  3. Create your own scenes with external backgrounds")


if __name__ == "__main__":
    main()
