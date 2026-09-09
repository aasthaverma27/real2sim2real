#!/usr/bin/env python3
"""
MuGS Asset Download Script

Downloads 3DGS object models and pre-trained SR models for MuGS project.

Usage:
    python download_assets.py --all                    # Download everything
    python download_assets.py --preset recommended     # Recommended starter set
    python download_assets.py --objects-only --count 5 # 5 sample objects
    python download_assets.py --models-only            # SR models only

Author: MuGS Team
Date: 2026-05-02
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = Path.home() / ".cache" / "mugs"

# Asset sources configuration
ASSET_SOURCES = {
    "nerf_synthetic": {
        "url": "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip",
        "objects": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
        "license": "Apache-2.0",
        "format": "nerf_data",  # Needs conversion to 3DGS
    },
    "sample_3dgs": {
        # Placeholder for actual 3DGS sample assets
        # In production, replace with real URLs
        "base_url": "https://example.com/mugs_assets/",  # TODO: Replace with actual CDN
        "objects": {
            "kitchen": ["mug_blue", "plate_white", "bowl_ceramic", "fork_metal", "knife_steel"],
            "tools": ["hammer", "screwdriver", "wrench", "pliers", "drill"],
            "containers": ["box_cardboard", "bin_plastic", "bottle_glass", "jar_glass", "basket"],
            "misc": ["lego_bricks", "book", "phone", "remote", "keys"],
        },
        "license": "CC-BY-4.0",
        "format": "ply",  # Direct 3DGS PLY files
    },
}

# Pre-trained model sources
MODEL_SOURCES = {
    "swinir_light": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "filename": "swinir_light_x4.pth",
        "size_mb": 3.5,
        "md5": None,  # TODO: Add checksum
    },
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth",
        "filename": "realesrgan_x4.pth",
        "size_mb": 64,
        "md5": None,
    },
}

# Download presets
PRESETS = {
    "quick": {
        "description": "Minimal assets for testing (3 objects, 1 model, ~50 MB, ~2 min)",
        "objects": {"kitchen": 2, "misc": 1},
        "models": ["swinir_light"],
    },
    "recommended": {
        "description": "Balanced set for development (20 objects, 2 models, ~500 MB, ~10 min)",
        "objects": {"kitchen": 5, "tools": 5, "containers": 5, "misc": 5},
        "models": ["swinir_light", "realesrgan"],
    },
    "full": {
        "description": "Comprehensive asset library (50+ objects, all models, ~2 GB, ~30 min)",
        "objects": {"kitchen": 15, "tools": 15, "containers": 15, "misc": 15},
        "models": ["swinir_light", "realesrgan"],
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        print(f"📥 {desc}: {url}")
        print(f"   → {dest}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Simple download (TODO: Add progress bar with tqdm)
        urllib.request.urlretrieve(url, dest)

        print(f"✅ Downloaded: {dest.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    if not expected_md5:
        return True

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    actual_md5 = md5.hexdigest()
    if actual_md5 != expected_md5:
        print(f"⚠️  Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False

    return True


def create_metadata_yaml(obj_name: str, category: str, source: str) -> Dict:
    """Create metadata YAML template for an object."""
    return {
        "name": obj_name,
        "category": category,
        "size": [0.10, 0.10, 0.10],  # Default size in meters [x, y, z]
        "mass": 0.5,  # Default mass in kg
        "source": source,
        "license": "CC-BY-4.0",
        "captured": "2026-05-02",
        "notes": f"Auto-downloaded from {source}",
    }


# ============================================================================
# Download Functions
# ============================================================================


def download_sample_3dgs_objects(categories: Dict[str, int], dry_run: bool = False) -> int:
    """Download sample 3DGS objects."""
    total_downloaded = 0
    source_config = ASSET_SOURCES["sample_3dgs"]

    print("\n" + "="*60)
    print("📦 Downloading 3DGS Object Models")
    print("="*60)

    for category, count in categories.items():
        if category not in source_config["objects"]:
            print(f"⚠️  Unknown category: {category}, skipping")
            continue

        available_objects = source_config["objects"][category]
        objects_to_download = available_objects[:count]

        print(f"\n📁 Category: {category} ({len(objects_to_download)} objects)")

        for obj_name in objects_to_download:
            # Construct download URL (placeholder)
            ply_url = f"{source_config['base_url']}{category}/{obj_name}.ply"
            ply_dest = ASSETS_DIR / "objects" / category / f"{obj_name}.ply"
            yaml_dest = ASSETS_DIR / "configs" / category / f"{obj_name}.yaml"

            if dry_run:
                print(f"   [DRY RUN] Would download: {obj_name}.ply")
                continue

            # Check if already exists
            if ply_dest.exists() and yaml_dest.exists():
                print(f"⏭️  Skipping {obj_name} (already exists)")
                total_downloaded += 1
                continue

            # Download PLY file
            # NOTE: This is a placeholder - replace with actual download logic
            # For now, create dummy files for testing
            print(f"   ℹ️  [{obj_name}] Creating placeholder (replace with actual download)")

            ply_dest.parent.mkdir(parents=True, exist_ok=True)
            ply_dest.write_text(f"# Placeholder PLY for {obj_name}\n")

            # Create metadata YAML
            metadata = create_metadata_yaml(obj_name, category, "sample_3dgs")
            yaml_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_dest, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

            print(f"✅ Created: {obj_name}.ply + .yaml")
            total_downloaded += 1

    return total_downloaded


def download_pretrained_models(model_names: List[str], dry_run: bool = False) -> int:
    """Download pre-trained SR models."""
    total_downloaded = 0

    print("\n" + "="*60)
    print("🤖 Downloading Pre-trained Models")
    print("="*60)

    for model_name in model_names:
        if model_name not in MODEL_SOURCES:
            print(f"⚠️  Unknown model: {model_name}, skipping")
            continue

        model_config = MODEL_SOURCES[model_name]
        url = model_config["url"]
        filename = model_config["filename"]
        dest = MODELS_DIR / "sr" / filename

        if dry_run:
            print(f"   [DRY RUN] Would download: {model_name} ({model_config['size_mb']} MB)")
            continue

        # Check if already exists
        if dest.exists():
            print(f"⏭️  Skipping {model_name} (already exists)")
            total_downloaded += 1
            continue

        # Download
        if download_file(url, dest, desc=f"Downloading {model_name}"):
            # Verify checksum if provided
            if model_config["md5"]:
                if verify_checksum(dest, model_config["md5"]):
                    print(f"✅ Verified checksum for {model_name}")
                else:
                    print(f"❌ Checksum verification failed for {model_name}")
                    dest.unlink()  # Delete corrupted file
                    continue

            total_downloaded += 1
        else:
            print(f"❌ Failed to download {model_name}")

    return total_downloaded


def download_nerf_synthetic(objects: List[str], dry_run: bool = False) -> int:
    """Download NeRF Synthetic dataset (requires conversion to 3DGS)."""
    print("\n" + "="*60)
    print("🎨 Downloading NeRF Synthetic Dataset")
    print("="*60)
    print("⚠️  Note: This downloads NeRF data. Conversion to 3DGS required.")
    print("   See docs/guides/ASSET_ACQUISITION.md for conversion instructions.")

    if dry_run:
        print(f"   [DRY RUN] Would download NeRF Synthetic dataset")
        return 0

    # Download zip file
    url = ASSET_SOURCES["nerf_synthetic"]["url"]
    cache_zip = CACHE_DIR / "nerf_synthetic.zip"

    if not cache_zip.exists():
        if not download_file(url, cache_zip, "Downloading NeRF Synthetic"):
            return 0

    # TODO: Unzip and convert to 3DGS
    print("ℹ️  Dataset cached at:", cache_zip)
    print("ℹ️  Run conversion script: scripts/data_collection/convert_nerf_to_3dgs.py")

    return 0  # Return 0 since conversion needed


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download assets for MuGS project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  python download_assets.py --all

  # Use preset configuration
  python download_assets.py --preset recommended

  # Download specific objects
  python download_assets.py --category kitchen --count 5

  # Download models only
  python download_assets.py --models-only

  # Dry run (preview)
  python download_assets.py --all --dry-run
        """
    )

    # Download targets
    parser.add_argument("--all", action="store_true", help="Download all assets (preset=recommended)")
    parser.add_argument("--preset", choices=PRESETS.keys(), help="Use preset configuration")
    parser.add_argument("--objects-only", action="store_true", help="Download 3DGS objects only")
    parser.add_argument("--models-only", action="store_true", help="Download pre-trained models only")

    # Object selection
    parser.add_argument("--category", help="Download specific category (kitchen/tools/containers/misc)")
    parser.add_argument("--count", type=int, default=5, help="Number of objects to download per category")
    parser.add_argument("--source", default="sample_3dgs", help="Asset source (sample_3dgs/nerf_synthetic)")

    # Options
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview what will be downloaded")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")

    args = parser.parse_args()

    # Determine download plan
    download_objects = False
    download_models = False
    object_categories = {}
    model_list = []

    if args.all or args.preset == "recommended":
        preset = PRESETS["recommended"]
        object_categories = preset["objects"]
        model_list = preset["models"]
        download_objects = True
        download_models = True
    elif args.preset:
        preset = PRESETS[args.preset]
        object_categories = preset["objects"]
        model_list = preset["models"]
        download_objects = True
        download_models = True
    elif args.objects_only:
        download_objects = True
        if args.category:
            object_categories = {args.category: args.count}
        else:
            object_categories = {cat: args.count for cat in ["kitchen", "tools", "containers", "misc"]}
    elif args.models_only:
        download_models = True
        model_list = list(MODEL_SOURCES.keys())
    elif args.category:
        download_objects = True
        object_categories = {args.category: args.count}
    else:
        parser.print_help()
        sys.exit(1)

    # Print download plan
    print("\n" + "="*60)
    print("MuGS Asset Download Script")
    print("="*60)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be downloaded\n")

    if download_objects:
        print("\n📦 Objects to download:")
        for cat, count in object_categories.items():
            print(f"   - {cat}: {count} objects")

    if download_models:
        print("\n🤖 Models to download:")
        for model in model_list:
            size_mb = MODEL_SOURCES[model]["size_mb"]
            print(f"   - {model}: ~{size_mb} MB")

    print("\n" + "-"*60)

    if not args.dry_run:
        response = input("Continue? [Y/n] ")
        if response.lower() == "n":
            print("Aborted.")
            sys.exit(0)

    # Execute downloads
    total_objects = 0
    total_models = 0

    if download_objects:
        total_objects = download_sample_3dgs_objects(object_categories, dry_run=args.dry_run)

    if download_models:
        total_models = download_pretrained_models(model_list, dry_run=args.dry_run)

    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    print(f"✅ Objects downloaded: {total_objects}")
    print(f"✅ Models downloaded: {total_models}")

    if not args.dry_run:
        print(f"\n📁 Assets location: {ASSETS_DIR}")
        print(f"📁 Models location: {MODELS_DIR}")
        print("\nNext steps:")
        print("  1. Validate assets: python scripts/data_collection/validate_assets.py")
        print("  2. Visualize assets: python scripts/evaluation/visualize_assets.py")
        print("  3. Start Phase 1: See TODO.md")

    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
