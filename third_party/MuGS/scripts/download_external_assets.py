#!/usr/bin/env python3
"""
Download external 3DGS assets from related projects.

Supported sources:
- GS-Playground (Bridge-GS dataset)
- DISCOVERSE (Scene assets)
- Other 3DGS benchmarks
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub not installed. Install with: pip install huggingface-hub")


ASSET_SOURCES = {
    "bridge-gs": {
        "repo_id": "YLab-Open/BRIDGE-Open",  # Bridge-GS dataset
        "files": [
            "README.md",
            # Add specific PLY files as they become available
        ],
        "description": "Bridge-GS: Large-scale manipulation dataset with 3DGS assets",
    },
    "discoverse-scenes": {
        "repo_id": "discoverse/scenes",  # Placeholder - actual repo TBD
        "files": [],
        "description": "DISCOVERSE scene assets with 3DGS backgrounds",
    },
    "interior-gs": {
        "repo_id": "spatialverse/InteriorGS",
        "files": [],
        "description": "InteriorGS: Indoor scenes with semantic labels",
    },
}


def check_git_lfs():
    """Check if Git LFS is installed."""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_from_huggingface(
    repo_id: str,
    output_dir: Path,
    files: Optional[List[str]] = None,
    use_git_lfs: bool = False,
) -> bool:
    """
    Download assets from Hugging Face.

    Args:
        repo_id: Hugging Face repository ID (e.g., "username/repo")
        output_dir: Local directory to save files
        files: Specific files to download (None = download all)
        use_git_lfs: Use git clone with LFS instead of HF API

    Returns:
        True if successful
    """
    if not HF_AVAILABLE and not use_git_lfs:
        print("❌ huggingface_hub not available and git-lfs not requested")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_git_lfs:
            # Use git clone with LFS
            if not check_git_lfs():
                print("❌ Git LFS not installed. Install with: git lfs install")
                return False

            repo_url = f"https://huggingface.co/datasets/{repo_id}"
            print(f"📥 Cloning {repo_url} with Git LFS...")

            result = subprocess.run(
                ["git", "clone", repo_url, str(output_dir)],
                check=True,
            )
            return result.returncode == 0
        else:
            # Use HF API
            if files:
                # Download specific files
                print(f"📥 Downloading {len(files)} files from {repo_id}...")
                for file in files:
                    print(f"  - {file}")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        local_dir=output_dir,
                        repo_type="dataset",
                    )
            else:
                # Download entire repository
                print(f"📥 Downloading entire repository {repo_id}...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=output_dir,
                    repo_type="dataset",
                )

            print(f"✅ Downloaded to {output_dir}")
            return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_discoverse_assets(output_dir: Path) -> bool:
    """
    Download DISCOVERSE demo assets.

    Note: DISCOVERSE automatically downloads PLY models on first run.
    This function provides manual download option.
    """
    print("=" * 80)
    print("DISCOVERSE Asset Download")
    print("=" * 80)
    print("\nNote: DISCOVERSE automatically downloads assets on first run.")
    print("To manually clone the repository:")
    print("\n  git clone https://github.com/TATP-233/DISCOVERSE")
    print("  cd DISCOVERSE")
    print("  git lfs install")
    print("  git lfs pull")
    print("\nAssets will be downloaded from Hugging Face when needed.")
    print("=" * 80)
    return True


def download_gs_playground_assets(output_dir: Path) -> bool:
    """
    Download GS-Playground example assets.
    """
    print("=" * 80)
    print("GS-Playground Asset Download")
    print("=" * 80)

    # Clone the repository
    repo_url = "https://github.com/discoverse-dev/gs_playground"
    repo_dir = output_dir / "gs_playground"

    if repo_dir.exists():
        print(f"⚠️  Repository already exists at {repo_dir}")
        return True

    print(f"📥 Cloning {repo_url}...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
        )
        print(f"✅ Cloned to {repo_dir}")

        # Check for Hugging Face assets mentioned in repo
        print("\n📋 Check the repository for links to Hugging Face assets:")
        print(f"   {repo_dir}/README.md")
        print("\nLarge-scale 3DGS assets may be released separately on Hugging Face.")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Clone failed: {e}")
        return False


def list_available_sources():
    """List all available asset sources."""
    print("=" * 80)
    print("Available Asset Sources")
    print("=" * 80)

    for name, info in ASSET_SOURCES.items():
        print(f"\n📦 {name}")
        print(f"   Repo: {info['repo_id']}")
        print(f"   Description: {info['description']}")
        if info['files']:
            print(f"   Files: {len(info['files'])} specified")

    print("\n" + "=" * 80)
    print("Special Sources")
    print("=" * 80)
    print("\n📦 gs-playground")
    print("   GitHub: discoverse-dev/gs_playground")
    print("   Description: GS-Playground framework and examples")

    print("\n📦 discoverse")
    print("   GitHub: TATP-233/DISCOVERSE")
    print("   Description: DISCOVERSE framework (auto-downloads assets)")


def main():
    parser = argparse.ArgumentParser(
        description="Download external 3DGS assets for MuGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "source",
        nargs="?",
        choices=list(ASSET_SOURCES.keys()) + ["gs-playground", "discoverse", "list"],
        help="Asset source to download (use 'list' to see all)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "external",
        help="Output directory for downloaded assets (default: data/external)",
    )

    parser.add_argument(
        "--use-git-lfs",
        action="store_true",
        help="Use git clone with LFS instead of HF API",
    )

    args = parser.parse_args()

    if args.source == "list" or args.source is None:
        list_available_sources()
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output}")

    # Handle special sources
    if args.source == "gs-playground":
        success = download_gs_playground_assets(args.output)
    elif args.source == "discoverse":
        success = download_discoverse_assets(args.output)
    else:
        # Handle standard Hugging Face sources
        source_info = ASSET_SOURCES[args.source]
        output_dir = args.output / args.source

        success = download_from_huggingface(
            repo_id=source_info["repo_id"],
            output_dir=output_dir,
            files=source_info["files"] if source_info["files"] else None,
            use_git_lfs=args.use_git_lfs,
        )

    if success:
        print("\n✅ Download completed successfully!")
        print(f"\nAssets saved to: {args.output}")
    else:
        print("\n❌ Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
