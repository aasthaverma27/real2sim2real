#!/usr/bin/env bash
# Offline MuGS render of a built scene to images / video (no robot stack).
# gsplat JIT-compiles CUDA kernels on first use, so the conda toolchain has to
# be on PATH (system gcc fails on conda's sysroot headers, and nvcc must be the
# conda wrapper that can find cicc).
#
#   bash scripts/run_mugs_render.sh --scene-dir outputs/myroom
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1
A="${RS_ENV_ASSET:-$(dirname "$(dirname "$RS_PY_ASSET")")}"   # supplies CUDA 12.8 + gcc 12.4
SIM="${RS_ENV_SIM:-$(dirname "$(dirname "$RS_PY_SIM")")}"     # has mujoco + gsplat
export CUDA_HOME="$A"
export PATH="$A/bin:$PATH"
export CC="$A/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$A/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
SITE="$("$SIM/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
export CPATH="$A/targets/x86_64-linux/include:$A/include:${CPATH:-}"
export LD_LIBRARY_PATH="$A/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/third_party/MuGS/src:${PYTHONPATH:-}"
exec "$SIM/bin/python" "$ROOT/scripts/mugs_render.py" "$@"
