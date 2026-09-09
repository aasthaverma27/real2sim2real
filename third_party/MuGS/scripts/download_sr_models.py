#!/usr/bin/env python3
"""
Download pretrained super-resolution models for MuGS.

Supports Real-ESRGAN models from official releases.
"""

import argparse
import sys
from pathlib import Path
import urllib.request
from typing import Dict


# Model definitions
MODELS: Dict[str, Dict[str, str]] = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size": "64 MB",
        "description": "General purpose, best quality for real photos",
    },
    "RealESRNet_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "size": "64 MB",
        "description": "Faster inference, slightly lower quality",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "size": "17 MB",
        "description": "Optimized for anime/cartoon images",
    },
}


def download_model(model_name: str, output_dir: Path, force: bool = False) -> bool:
    """
    Download a pretrained model.

    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
        force: Force re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    url = model_info["url"]
    filename = f"{model_name}.pth"
    output_path = output_dir / filename

    # Check if already exists
    if output_path.exists() and not force:
        print(f"✅ Model already exists: {output_path}")
        print(f"   Use --force to re-download")
        return True

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {model_name}...")
    print(f"   URL: {url}")
    print(f"   Size: {model_info['size']}")
    print(f"   Destination: {output_path}")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            bar_length = 50
            filled = int(bar_length * percent // 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r   Progress: [{bar}] {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()  # New line after progress bar

        # Verify file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Downloaded successfully: {file_size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def list_models():
    """List all available models."""
    print("=" * 80)
    print("Available Super-Resolution Models")
    print("=" * 80)

    for model_name, info in MODELS.items():
        print(f"\n{model_name}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")

    print("\n" + "=" * 80)
    print("Download with: python scripts/download_sr_models.py --model MODEL_NAME")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained super-resolution models for MuGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download recommended model
  python scripts/download_sr_models.py --model RealESRGAN_x4plus

  # Download all models
  python scripts/download_sr_models.py --all

  # List available models
  python scripts/download_sr_models.py --list

  # Force re-download
  python scripts/download_sr_models.py --model RealESRGAN_x4plus --force
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Model to download"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pretrained/sr"),
        help="Output directory (default: data/pretrained/sr)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return

    # Download all models
    if args.all:
        print("=" * 80)
        print("Downloading All Super-Resolution Models")
        print("=" * 80)

        success_count = 0
        for model_name in MODELS.keys():
            print(f"\n{'-' * 80}")
            if download_model(model_name, args.output_dir, args.force):
                success_count += 1

        print(f"\n{'=' * 80}")
        print(f"✅ Downloaded {success_count}/{len(MODELS)} models successfully")
        print(f"{'=' * 80}")
        return

    # Download specific model
    if args.model:
        print("=" * 80)
        print("Downloading Super-Resolution Model")
        print("=" * 80)

        if download_model(args.model, args.output_dir, args.force):
            print("\n" + "=" * 80)
            print("✅ Download complete!")
            print("=" * 80)
            print("\nUsage:")
            print(f"  from mugs.postprocess import SuperResolution, SuperResolutionConfig")
            print(f"  sr = SuperResolution(SuperResolutionConfig(model_name='{args.model}'))")
            print(f"  img_hr = sr.upscale(img_lr)")
        else:
            sys.exit(1)
        return

    # No arguments - show help
    parser.print_help()
    print("\n" + "=" * 80)
    print("Quick Start:")
    print("=" * 80)
    print("\n1. Download the recommended model:")
    print("     python scripts/download_sr_models.py --model RealESRGAN_x4plus")
    print("\n2. Use in your code:")
    print("     from mugs.postprocess import SuperResolution, SuperResolutionConfig")
    print("     sr = SuperResolution(SuperResolutionConfig())")
    print("     img_hr = sr.upscale(img_lr)")


if __name__ == "__main__":
    main()
