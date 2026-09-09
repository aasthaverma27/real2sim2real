#!/usr/bin/env python3
"""
Download Mip-NeRF 360 Dataset

Downloads and prepares Mip-NeRF 360 scenes for 3DGS training.

Usage:
    python download_mipnerf360.py --scene kitchen
    python download_mipnerf360.py --scene all

Author: MuGS Team
Date: 2026-05-02
"""

import argparse
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"📥 Downloading: {url}")
    print(f"   → {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

    try:
        urlretrieve(url, output_path, reporthook=report_progress)
        print()  # New line after progress
        print(f"✅ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    print(f"\n📦 Extracting: {zip_path.name}")
    print(f"   → {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size
            total_size = sum(info.file_size for info in zip_ref.filelist)
            extracted_size = 0

            for info in zip_ref.filelist:
                zip_ref.extract(info, extract_to)
                extracted_size += info.file_size
                percent = extracted_size * 100 / total_size
                print(f"\r   Progress: {percent:.1f}%", end='')

            print()

        print(f"✅ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False


def download_mipnerf360(output_dir: Path, scenes: list = None):
    """Download Mip-NeRF 360 dataset."""
    print("="*70)
    print("Mip-NeRF 360 Dataset Downloader")
    print("="*70)
    print()

    # Dataset URL
    dataset_url = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download path
    zip_path = output_dir / "360_v2.zip"

    # Download if not exists
    if zip_path.exists():
        print(f"⏭️  Zip already exists: {zip_path}")
        print(f"   Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    else:
        if not download_with_progress(dataset_url, zip_path):
            return False

    # Extract
    extract_dir = output_dir / "360_v2"
    if extract_dir.exists():
        print(f"\n⏭️  Already extracted: {extract_dir}")
    else:
        if not extract_zip(zip_path, output_dir):
            return False

    # List available scenes
    print(f"\n📁 Available scenes in {extract_dir}:")
    available_scenes = []
    for scene_dir in sorted(extract_dir.iterdir()):
        if scene_dir.is_dir():
            # Count images
            image_dir = scene_dir / "images"
            if image_dir.exists():
                num_images = len(list(image_dir.glob("*.JPG"))) + len(list(image_dir.glob("*.jpg")))
                size_mb = sum(f.stat().st_size for f in scene_dir.rglob("*") if f.is_file()) / (1024**2)
                print(f"   - {scene_dir.name}: {num_images} images, {size_mb:.1f} MB")
                available_scenes.append(scene_dir.name)

    # Filter scenes if specified
    if scenes and scenes != ['all']:
        print(f"\n🎯 Requested scenes: {scenes}")
        missing = set(scenes) - set(available_scenes)
        if missing:
            print(f"   ⚠️  Not found: {missing}")
        scenes_to_use = [s for s in scenes if s in available_scenes]
    else:
        scenes_to_use = available_scenes

    print(f"\n✅ Downloaded {len(available_scenes)} scenes")
    print(f"   Total size: {zip_path.stat().st_size / (1024**3):.2f} GB (zip)")

    # Print next steps
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print()
    print("Option 1: Train 3DGS yourself (recommended for customization)")
    print("-" * 70)
    print("# Install 3DGS trainer")
    print("git clone https://github.com/graphdeco-inria/gaussian-splatting")
    print("cd gaussian-splatting")
    print("pip install -r requirements.txt")
    print()
    print("# Train kitchen scene (~1-2 hours on RTX 4090)")
    print(f"python train.py -s {extract_dir / 'kitchen'} -m output/kitchen")
    print()
    print("# Copy trained model")
    print("cp output/kitchen/point_cloud/iteration_30000/point_cloud.ply \\")
    print("   /home/ununtu/metabot-workspace/mugs/data/pretrained/kitchen/point_cloud/iteration_30000/point_cloud.ply")
    print()
    print()
    print("Option 2: Download pre-trained models (faster)")
    print("-" * 70)
    print("# Search for pre-trained Mip-NeRF 360 3DGS models")
    print("# https://huggingface.co/models?search=gaussian+splatting+mipnerf360")
    print()
    print("# Or check GitHub for shared models")
    print("# https://github.com/topics/3d-gaussian-splatting")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Mip-NeRF 360 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all scenes
  python download_mipnerf360.py --scene all

  # Download only kitchen
  python download_mipnerf360.py --scene kitchen

  # Download multiple scenes
  python download_mipnerf360.py --scene kitchen --scene room
        """
    )

    parser.add_argument(
        '--scene',
        action='append',
        default=[],
        help='Scene name to download (kitchen, room, counter, etc). Use "all" for all scenes.'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "data",
        help='Output directory (default: PROJECT_ROOT/data/)'
    )

    args = parser.parse_args()

    if not args.scene:
        args.scene = ['all']

    print(f"Output directory: {args.output_dir}")
    print(f"Scenes: {args.scene}")
    print()

    success = download_mipnerf360(args.output_dir, args.scene)

    if success:
        print("\n✅ Download complete!")
    else:
        print("\n❌ Download failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
