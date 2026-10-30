#!/usr/bin/env bash
# Compile gsplat's CUDA kernels once, in the foreground.
#
# gsplat builds its kernels the first time anything renders a splat, and that
# takes several minutes on a fresh machine. start.sh runs this before opening
# any window, so the renderers find the kernels ready instead of each showing
# a black window while they compile. Once built, it returns in a few seconds.
#
#   scripts/build_gsplat.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1

echo "[gsplat] CUDA kernels for GPU arch ${TORCH_CUDA_ARCH_LIST:-all}: building if needed (first run takes a few minutes)"
t0=$(date +%s)
"$RS_PY_SIM" -c "from gsplat.cuda._backend import _C; assert _C is not None" 2>&1 \
  | grep -v "TORCH_CUDA_ARCH_LIST\|warnings.warn\|If this is not desired" || true
"$RS_PY_SIM" -c "from gsplat.cuda._backend import _C; assert _C is not None" >/dev/null 2>&1 \
  || { echo "[gsplat] build failed - run: $RS_PY_SIM -c 'from gsplat.cuda._backend import _C'" >&2; exit 1; }
echo "[gsplat] kernels ready ($(( $(date +%s) - t0 )) s)"
