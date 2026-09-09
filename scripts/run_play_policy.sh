#!/usr/bin/env bash
# Batched photoreal env: warp physics + warp foreground + batched gsplat background.
set -euo pipefail
ROOT="${ROBOSNAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
A=${RS_ENV_ASSET:-/mnt/ssd2/conda/envs/robosnap-asset}
SIM=${RS_ENV_SIM:-/mnt/ssd2/conda/envs/robosnap-sim}
export CUDA_HOME="$A"
export PATH="$A/bin:$PATH"
export CC="$A/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$A/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
SITE="$("$SIM/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
export CPATH="$A/targets/x86_64-linux/include:$A/include:${CPATH:-}"
export LD_LIBRARY_PATH="$A/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$SIM/bin/python" "$ROOT/scripts/play_policy.py" "$@"
