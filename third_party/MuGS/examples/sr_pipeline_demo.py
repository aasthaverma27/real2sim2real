#!/usr/bin/env python3
"""
Super-Resolution Pipeline Demo

Demonstrates:
1. Low-resolution rendering for speed
2. Optional super-resolution upscaling for quality
3. Comparison visualization
4. Batch processing

Prerequisites:
    1. Install dependencies:
       pip install realesrgan basicsr

    2. Download SR models:
       python scripts/download_sr_models.py --model RealESRGAN_x4plus
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

from mugs.sensors import GaussianSensor, GaussianSensorConfig

# Try to import SR module (optional)
try:
    from mugs.postprocess import SuperResolution, SuperResolutionConfig
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("⚠️  Super-resolution not available (missing dependencies)")


def create_simple_scene():
    """Create a simple YAM-like scene for demo."""
    xml = """
    <mujoco model="sr_demo">
        <compiler angle="radian"/>

        <visual>
            <global offwidth="640" offheight="480"/>
            <quality shadowsize="4096"/>
            <headlight ambient="0.5 0.5 0.5"/>
        </visual>

        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
        </asset>

        <worldbody>
            <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 3"/>
            <camera name="main_camera" pos="1.5 -1.5 1.2" xyaxes="0.707 0.707 0 -0.408 0.408 0.816" fovy="0.785"/>

            <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>

            <!-- Robot arm (simplified YAM) -->
            <body name="base" pos="0 0 0.1">
                <geom name="base_link" type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.3 1"/>
                <body name="link1" pos="0 0 0.15">
                    <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                    <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.3" size="0.04" rgba="0.5 0.5 0.5 1"/>
                    <body name="link2" pos="0 0 0.3">
                        <joint name="j2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom name="link2_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.035" rgba="0.6 0.6 0.6 1"/>
                        <body name="gripper" pos="0.3 0 0">
                            <geom name="gripper_geom" type="box" size="0.05 0.02 0.02" rgba="0.7 0.7 0.7 1"/>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Object (red cube) -->
            <body name="cube" pos="0.4 0.2 0.12">
                <joint type="free"/>
                <geom name="cube_geom" type="box" size="0.04 0.04 0.04" rgba="0.9 0.2 0.2 1" mass="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Save to temporary file
    with NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml)
        xml_path = f.name

    return xml_path


def demo_basic_sr():
    """Demo 1: Basic super-resolution usage."""
    print("\n" + "=" * 80)
    print("Demo 1: Basic Super-Resolution")
    print("=" * 80)

    # Create scene
    xml_path = create_simple_scene()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Low-res sensor
    sensor_config = GaussianSensorConfig(
        width=320,
        height=240,
        background_ply_path=Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"),
        render_mode="hybrid",
        robot_geom_names=["base_link", "link1_geom", "link2_geom", "gripper_geom", "cube_geom"],
    )
    sensor = GaussianSensor(sensor_config)

    # Simulate
    mujoco.mj_forward(model, data)

    # Render low-res
    print("\n📊 Rendering low-resolution image (320×240)...")
    img_lr = sensor.render(model, data, "main_camera", return_components=True)

    print(f"✅ Low-res rendered: {img_lr['rgb'].shape}")

    # Super-resolution
    if not SR_AVAILABLE:
        print("\n⚠️  Super-resolution not available")
        print("   Install with: pip install realesrgan basicsr")
        print("   Download models: python scripts/download_sr_models.py --model RealESRGAN_x4plus")
        img_hr = None
    else:
        try:
            print("\n🚀 Upscaling with Real-ESRGAN (4x)...")
            sr = SuperResolution(SuperResolutionConfig(
                model_name="RealESRGAN_x4plus",
                scale=4,
            ))
            img_hr = sr.upscale(img_lr['rgb'])
            print(f"✅ High-res upscaled: {img_hr.shape}")
        except Exception as e:
            print(f"\n❌ Super-resolution failed: {e}")
            print("   Make sure models are downloaded:")
            print("   python scripts/download_sr_models.py --model RealESRGAN_x4plus")
            img_hr = None

    # Visualize
    print("\n📊 Creating visualization...")
    if img_hr is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Comparison
        axes[0, 0].imshow(img_lr['rgb'])
        axes[0, 0].set_title('Low-Res (320×240)\nFast rendering', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Simple upscale for comparison
        import cv2
        img_simple = cv2.resize(img_lr['rgb'], (img_hr.shape[1], img_hr.shape[0]), interpolation=cv2.INTER_LINEAR)
        axes[0, 1].imshow(img_simple)
        axes[0, 1].set_title('Bilinear Upscale (1280×960)\nSimple interpolation', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img_hr)
        axes[0, 2].set_title('Real-ESRGAN 4x (1280×960)\nAI upscaling ✨', fontsize=14, fontweight='bold', color='#d62728')
        axes[0, 2].axis('off')

        # Row 2: Details (cropped)
        h, w = img_lr['rgb'].shape[:2]
        crop_lr = img_lr['rgb'][h//3:2*h//3, w//3:2*w//3]

        h_hr, w_hr = img_hr.shape[:2]
        crop_simple = img_simple[h_hr//3:2*h_hr//3, w_hr//3:2*w_hr//3]
        crop_hr = img_hr[h_hr//3:2*h_hr//3, w_hr//3:2*w_hr//3]

        axes[1, 0].imshow(crop_lr)
        axes[1, 0].set_title('Detail: Low-Res\nPixelated', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(crop_simple)
        axes[1, 1].set_title('Detail: Bilinear\nBlurry', fontsize=12)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(crop_hr)
        axes[1, 2].set_title('Detail: Real-ESRGAN\nSharp & realistic', fontsize=12, color='#d62728')
        axes[1, 2].axis('off')

        plt.suptitle('MuGS + Super-Resolution Pipeline Demo', fontsize=16, fontweight='bold')
        plt.tight_layout()
    else:
        # Fallback: show only low-res
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_lr['rgb'])
        ax.set_title('Low-Res Rendering (320×240)\nSuper-resolution not available', fontsize=14, fontweight='bold')
        ax.axis('off')

    # Save
    output_dir = Path("outputs/sr_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sr_comparison.jpg"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Copy to user output
    import shutil
    user_output = Path("/tmp/metabot-outputs-ununtu/oc_70a36f8040cc57178505765cfa3ae250/sr_comparison.jpg")
    if user_output.parent.exists():
        shutil.copy(output_path, user_output)
        print(f"✅ Copied to: {user_output}")

    plt.close()

    # Cleanup
    Path(xml_path).unlink()


def demo_batch_processing():
    """Demo 2: Batch processing with SR."""
    print("\n" + "=" * 80)
    print("Demo 2: Batch Processing")
    print("=" * 80)

    if not SR_AVAILABLE:
        print("⚠️  Super-resolution not available, skipping batch demo")
        return

    # Create scene
    xml_path = create_simple_scene()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Sensor
    sensor_config = GaussianSensorConfig(
        width=160,
        height=120,
        background_ply_path=Path("data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply"),
        render_mode="hybrid",
        robot_geom_names=["base_link", "link1_geom", "link2_geom", "gripper_geom", "cube_geom"],
    )
    sensor = GaussianSensor(sensor_config)

    # SR
    try:
        sr = SuperResolution(SuperResolutionConfig(
            model_name="RealESRGAN_x4plus",
            scale=4,
        ))
    except Exception as e:
        print(f"❌ Cannot create SR module: {e}")
        Path(xml_path).unlink()
        return

    # Render sequence
    print("\n📊 Rendering sequence (10 frames)...")
    imgs_lr = []
    for i in range(10):
        # Animate joint
        data.qpos[0] = i * 0.3
        mujoco.mj_forward(model, data)

        img = sensor.render(model, data, "main_camera")
        imgs_lr.append(img)

    print(f"✅ Rendered {len(imgs_lr)} frames at 160×120")

    # Batch upscale
    print("\n🚀 Batch upscaling...")
    imgs_hr = sr.batch_upscale(imgs_lr, show_progress=True)

    print(f"✅ Upscaled to 640×480")

    # Save first and last frames
    output_dir = Path("outputs/sr_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(imgs_lr[0])
    axes[0, 0].set_title('Frame 0: Low-Res', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(imgs_hr[0])
    axes[0, 1].set_title('Frame 0: Super-Res', fontsize=12)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(imgs_lr[-1])
    axes[1, 0].set_title('Frame 9: Low-Res', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(imgs_hr[-1])
    axes[1, 1].set_title('Frame 9: Super-Res', fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle('Batch Super-Resolution Demo', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "batch_sr.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Cleanup
    Path(xml_path).unlink()


def main():
    print("=" * 80)
    print("MuGS Super-Resolution Pipeline Demo")
    print("=" * 80)

    # Check dependencies
    print("\n📋 Checking dependencies...")

    try:
        import realesrgan
        import basicsr
        print("✅ Real-ESRGAN installed")
    except ImportError:
        print("❌ Real-ESRGAN not installed")
        print("   Install with: pip install realesrgan basicsr")

    # Check models
    model_path = Path("data/pretrained/sr/RealESRGAN_x4plus.pth")
    if model_path.exists():
        print(f"✅ SR model found: {model_path}")
    else:
        print(f"❌ SR model not found: {model_path}")
        print("   Download with: python scripts/download_sr_models.py --model RealESRGAN_x4plus")

    # Run demos
    demo_basic_sr()
    demo_batch_processing()

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)

    print("\n📚 Next steps:")
    print("  1. Check outputs in: outputs/sr_demo/")
    print("  2. Read docs: docs/SUPER_RESOLUTION.md")
    print("  3. Download more models: python scripts/download_sr_models.py --list")


if __name__ == "__main__":
    main()
