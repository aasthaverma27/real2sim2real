#!/usr/bin/env bash
# =============================================================================
# Download DISCOVERSE 3DGS room scenes and unpack the SuperSplat-packed PLYs
# into the standard 3DGS layout consumed by ``mugs.sensors.GaussianSensor``.
#
# Pipeline:
#   1. Pull packed PLY(s) from ``tatp/DISCOVERSE-models`` on Hugging Face
#      (path: ``3dgs/scene/<name>/point_cloud.ply``).
#   2. Decompress via ``scripts/data_collection/decompress_supersplat.py``
#      into the standard ``x/y/z + f_dc_* + scale_* + rot_* + opacity`` schema.
#
# By design no scenes are tracked in git (see .gitignore — ``assets/scenes/*.ply``
# is ignored). This script is the only source of truth for fetching them.
#
# Usage:
#   bash scripts/data_collection/download_discoverse_scenes.sh                    # default: lab3
#   bash scripts/data_collection/download_discoverse_scenes.sh --all              # all 4 rooms
#   bash scripts/data_collection/download_discoverse_scenes.sh lab3 flower_table  # subset
#   bash scripts/data_collection/download_discoverse_scenes.sh --dst-root <dir>   # custom output
#
# Env:
#   DISCOVERSE_SCENE_ROOT    override DEFAULT_DST_ROOT (default: <repo>/assets/scenes/discoverse)
#
# Exit codes: 0 ok / 1 missing dep / 2 download fail / 3 decompress fail.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HF_REPO="tatp/DISCOVERSE-models"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main/3dgs/scene"

# All scenes confirmed present in the upstream HF repo as of 2026-05.
ALL_SCENES=(lab3 flower_table discover_operation_studio tsimf_library_0)

DST_ROOT="${DISCOVERSE_SCENE_ROOT:-${REPO_ROOT}/assets/scenes/discoverse}"
SCENES=()
WANT_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)        WANT_ALL=1; shift ;;
    --dst-root)   DST_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,26p' "$0"; exit 0 ;;
    -*)
      echo "[ERR] unknown flag: $1" >&2; exit 1 ;;
    *)
      SCENES+=("$1"); shift ;;
  esac
done

if [[ "$WANT_ALL" -eq 1 ]]; then
  SCENES=("${ALL_SCENES[@]}")
elif [[ "${#SCENES[@]}" -eq 0 ]]; then
  SCENES=(lab3)
fi

DECOMPRESS="${SCRIPT_DIR}/decompress_supersplat.py"
if [[ ! -f "$DECOMPRESS" ]]; then
  echo "[ERR] decompressor not found at: $DECOMPRESS" >&2
  exit 1
fi

# Pick the downloader: curl preferred, wget fallback.
if command -v curl >/dev/null 2>&1; then
  DL() { curl -fL --retry 3 --retry-delay 2 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  DL() { wget -q --tries=3 -O "$2" "$1"; }
else
  echo "[ERR] need curl or wget on PATH" >&2; exit 1
fi

PY="$(command -v python3 || command -v python)"
if [[ -z "$PY" ]]; then
  echo "[ERR] no usable python on PATH" >&2; exit 1
fi

PACKED_ROOT="${DST_ROOT}"
UNPACKED_ROOT="${DST_ROOT}_unpacked"

mkdir -p "$PACKED_ROOT" "$UNPACKED_ROOT"

echo "[info] scenes:    ${SCENES[*]}"
echo "[info] packed →   $PACKED_ROOT"
echo "[info] unpacked → $UNPACKED_ROOT"

for scene in "${SCENES[@]}"; do
  src_url="${HF_BASE}/${scene}/point_cloud.ply"
  packed_dir="${PACKED_ROOT}/${scene}"
  packed_ply="${packed_dir}/point_cloud.ply"
  unpacked_dir="${UNPACKED_ROOT}/${scene}"
  unpacked_ply="${unpacked_dir}/point_cloud.ply"

  mkdir -p "$packed_dir" "$unpacked_dir"

  if [[ -s "$packed_ply" ]]; then
    echo "[skip] packed exists: $packed_ply"
  else
    echo "[pull] $src_url"
    if ! DL "$src_url" "$packed_ply"; then
      echo "[ERR] download failed: $scene" >&2
      rm -f "$packed_ply"
      exit 2
    fi
  fi

  if [[ -s "$unpacked_ply" ]]; then
    echo "[skip] unpacked exists: $unpacked_ply"
  else
    echo "[unpack] $packed_ply → $unpacked_ply"
    if ! "$PY" "$DECOMPRESS" "$packed_ply" "$unpacked_ply"; then
      echo "[ERR] decompress failed: $scene" >&2
      exit 3
    fi
  fi
done

echo "[done] ${#SCENES[@]} scene(s) ready."
echo
echo "Render with mugs.sensors.GaussianSensor:"
echo "  background_ply_path=\"${UNPACKED_ROOT}/lab3/point_cloud.ply\""
