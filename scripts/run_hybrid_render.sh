#!/usr/bin/env bash
# Hybrid MuJoCo + 3DGS render. gsplat JIT-compiles CUDA kernels on first use, so
# the conda toolchain must be on the environment (system gcc fails on conda's
# sysroot headers; nvcc must be the conda wrapper that can find cicc).
set -euo pipefail
ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
A=${RS_ENV_ASSET:-/mnt/ssd2/conda/envs/robosnap-asset}          # supplies CUDA 12.8 + gcc 12.4
SIM=${RS_ENV_SIM:-/mnt/ssd2/conda/envs/robosnap-sim}          # has mujoco + gsplat
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
exec "$SIM/bin/python" "$ROOT/scripts/mujoco_splat_render.py" "$@"
