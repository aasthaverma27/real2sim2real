# Sourced by every shell entry point. Not executable, not run directly.
#
#     ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#     . "$ROOT/scripts/_bootstrap.sh"
#
# Finds the repo, loads configs/paths.env, and puts the simulation env's bin on
# PATH so gsplat's CUDA JIT can find its toolchain. The point is that a user
# never has to `source` anything by hand or remember which interpreter a script
# needs - reading the README and running the command is enough.

# shellcheck disable=SC2155
RS_ROOT="${RS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RS_ROOT

if [ -f "$RS_ROOT/configs/paths.env" ]; then
    # shellcheck disable=SC1091
    . "$RS_ROOT/configs/paths.env"
else
    cat >&2 <<EOF
configs/paths.env is missing.

It is normally written for you by the installer:

    bash scripts/install_auto_pipeline.sh

If your environments already exist and you only need the file, copy the
template and point it at them:

    cp configs/paths.env.example configs/paths.env
    \$EDITOR configs/paths.env

Then check everything with:  scripts/doctor.sh
EOF
    return 1 2>/dev/null || exit 1
fi

if [ -z "${RS_PY_SIM:-}" ] || [ ! -x "${RS_PY_SIM}" ]; then
    cat >&2 <<EOF
RS_PY_SIM does not point at a usable interpreter:
    ${RS_PY_SIM:-<unset>}

That environment holds mujoco and gsplat. Build it with:

    bash scripts/install_auto_pipeline.sh --only sim

Then check everything with:  scripts/doctor.sh
EOF
    return 1 2>/dev/null || exit 1
fi

# gsplat JIT-compiles CUDA kernels on first use and needs ninja plus the
# toolchain from this env; without it the first render dies with
# "Ninja is required to load C++ extensions".
case ":$PATH:" in
    *":$(dirname "$RS_PY_SIM"):"*) ;;
    *) export PATH="$(dirname "$RS_PY_SIM"):$PATH" ;;
esac

# Build those kernels for this machine's GPU only. Unset, torch compiles for
# every architecture it knows, which makes the first build many times slower.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$_cc" ] && export TORCH_CUDA_ARCH_LIST="$_cc"
    unset _cc
fi

export PYTHONPATH="$RS_ROOT:$RS_ROOT/third_party/MuGS/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DISPLAY="${DISPLAY:-:0}"
