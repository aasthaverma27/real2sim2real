#!/usr/bin/env bash
# MuGS GaussianSensor render of a RoboSnap scene. Same environment contract as
# run_hybrid_render.sh: gsplat JIT-compiles CUDA kernels on first use, so the
# conda toolchain has to be on PATH (system gcc fails on conda's sysroot
# headers, and nvcc must be the conda wrapper that can find cicc).
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/third_party/MuGS/src:${RS_GROOT_ROOT:-/mnt/ssd2/Isaac-GR00T}:${PYTHONPATH:-}"
exec "$SIM/bin/python" "$ROOT/scripts/play_groot_wbc.py" "$@"
