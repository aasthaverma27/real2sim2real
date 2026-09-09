#!/usr/bin/env python3
"""
MuGS Environment Check Script

Verifies that all required dependencies and configurations are properly installed.

Usage:
    python check_environment.py

Author: MuGS Team
Date: 2026-05-02
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("="*60)
    print("🐍 Python Version")
    print("="*60)

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Version: {version_str}")

    if version.major == 3 and version.minor in [10, 11, 12]:
        print("✅ Python version OK")
        return True
    else:
        print(f"⚠️  Recommended: Python 3.10, 3.11, or 3.12 (you have {version_str})")
        return False


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "="*60)
    print("🔥 PyTorch")
    print("="*60)

    try:
        import torch
        print(f"Version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

            # Check GPU memory
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {gpu_mem_gb:.1f} GB")

            if gpu_mem_gb >= 8:
                print("✅ PyTorch + CUDA OK")
                return True
            else:
                print("⚠️  GPU has <8GB VRAM, may have memory issues")
                return False
        else:
            print("❌ CUDA not available - GPU required for MuGS")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install: pip install torch torchvision")
        return False


def check_gsplat():
    """Check gsplat installation."""
    print("\n" + "="*60)
    print("🌟 gsplat")
    print("="*60)

    try:
        import gsplat
        print(f"Version: {gsplat.__version__}")
        print("✅ gsplat OK")
        return True
    except ImportError:
        print("❌ gsplat not installed")
        print("   Install: pip install gsplat")
        return False


def check_mujoco():
    """Check MuJoCo installation."""
    print("\n" + "="*60)
    print("🤖 MuJoCo")
    print("="*60)

    try:
        import mujoco
        print(f"Version: {mujoco.__version__}")
        print("✅ MuJoCo OK")
        return True
    except ImportError:
        print("⚠️  MuJoCo not installed (optional for Phase 1)")
        print("   Install: pip install mujoco")
        return True  # Not critical for Phase 1


def check_mugs():
    """Check MuGS package installation."""
    print("\n" + "="*60)
    print("📦 MuGS Package")
    print("="*60)

    try:
        import mugs
        print(f"Version: {mugs.__version__}")
        print("✅ MuGS package OK")
        return True
    except ImportError:
        print("❌ MuGS package not installed")
        print("   Install: pip install -e .")
        return False


def check_optional_deps():
    """Check optional dependencies."""
    print("\n" + "="*60)
    print("🔧 Optional Dependencies")
    print("="*60)

    deps = {
        "lpips": "Perceptual loss (for SR evaluation)",
        "opencv-python": "Image processing",
        "yaml": "Configuration files",
        "numpy": "Numerical computing",
        "matplotlib": "Visualization",
    }

    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module if module != "opencv-python" else "cv2")
            print(f"✅ {module}: {desc}")
        except ImportError:
            print(f"⚠️  {module}: {desc} (optional)")
            all_ok = False

    return all_ok


def check_assets():
    """Check if assets are downloaded."""
    print("\n" + "="*60)
    print("📁 Assets")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent
    assets_dir = project_root / "assets" / "objects"
    models_dir = project_root / "models" / "sr"

    # Count objects
    if assets_dir.exists():
        ply_files = list(assets_dir.rglob("*.ply"))
        yaml_files = list(assets_dir.rglob("*.yaml"))
        print(f"3DGS objects: {len(ply_files)} PLY files")
        print(f"Metadata: {len(yaml_files)} YAML files")
    else:
        print("⚠️  No assets directory found")
        ply_files = []

    # Count models
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.pt"))
        print(f"SR models: {len(model_files)} model files")
    else:
        print("⚠️  No models directory found")
        model_files = []

    if len(ply_files) > 0:
        print("✅ Assets found")
        return True
    else:
        print("⚠️  No assets downloaded")
        print("   Run: python scripts/data_collection/download_assets.py --preset quick")
        return True  # Not critical for initial setup


def main():
    """Run all environment checks."""
    print("\n" + "="*60)
    print("MuGS Environment Check")
    print("="*60)
    print()

    checks = [
        ("Python version", check_python_version),
        ("PyTorch + CUDA", check_pytorch),
        ("gsplat", check_gsplat),
        ("MuJoCo", check_mujoco),
        ("MuGS package", check_mugs),
        ("Optional dependencies", check_optional_deps),
        ("Assets", check_assets),
    ]

    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()

    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)

    critical_checks = ["Python version", "PyTorch + CUDA", "MuGS package"]
    critical_passed = all(results[name] for name in critical_checks if name in results)

    total = len(results)
    passed = sum(results.values())

    print(f"Checks passed: {passed}/{total}")
    print()

    if critical_passed:
        print("✅ All critical checks passed!")
        print("🚀 Ready to start Phase 1 development")
    else:
        print("❌ Some critical checks failed")
        print("   Please install missing dependencies before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
