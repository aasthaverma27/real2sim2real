#!/usr/bin/env bash
# Fetch the external stacks needed to run a VLA policy or a deployed whole-body
# controller, and say plainly which ones you can actually get.
#
#   scripts/setup_external.sh            fetch what is reachable, report the rest
#   scripts/setup_external.sh --check    report only, change nothing
#
# These are SUBMODULES, not vendored copies. That is a licence decision, not a
# style one: the whole-body controller is under the NVIDIA License, which limits
# use to non-commercial research (section 3.3) and attaches conditions to any
# redistribution (section 3.1). A submodule records a URL and a commit, so this
# repository distributes none of that code and inherits none of those terms.
# Do not replace these with checked-in copies.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

row() { printf '  %-30s %-10s %s\n' "$1" "$2" "$3"; }

echo "external dependencies"
echo "---------------------"
row "component" "state" "notes"

declare -A NEED=(
  [external/Isaac-GR00T]="policy server (Apache-2.0, public)"
  [external/unitree_sdk2_python]="DDS bridge (public)"
  [external/vla_for_simulation_testing]="sim bridge + WBC container (private; NVIDIA License inside)"
  [external/simple_wbc_code]="balance/walk ONNX + keyboard teleop (third-party)"
)

MISSING=()
for p in "${!NEED[@]}"; do
  if [ -n "$(ls -A "$p" 2>/dev/null)" ]; then
    row "$(basename "$p")" "present" "${NEED[$p]}"
  else
    MISSING+=("$p")
    row "$(basename "$p")" "MISSING" "${NEED[$p]}"
  fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
  echo; echo "all present."
  exit 0
fi

if [ "$CHECK" = 1 ]; then
  echo; echo "run without --check to attempt fetching them."
  exit 0
fi

echo
for p in "${MISSING[@]}"; do
  echo "[setup] fetching $p"
  if git submodule update --init --recursive "$p" 2>&1 | sed 's/^/    /'; then
    :
  else
    echo "    could not fetch $p"
  fi
done

echo
echo "still missing after fetch:"
STILL=0
for p in "${MISSING[@]}"; do
  [ -n "$(ls -A "$p" 2>/dev/null)" ] || { echo "  - $p"; STILL=1; }
done
[ "$STILL" = 0 ] && echo "  (none)"

cat <<'EOF'

What you can do without the private ones
----------------------------------------
Reconstruction, the MuJoCo scene, the photoreal hybrid, the robot cameras and
the BUNDLED controller all work with no external stack at all:

    scripts/realsim/start.sh --builtin

That runs the balance/walk ONNX nets in-process. It needs only the two .onnx
files - point RS_WBC_MODELS at them in configs/paths.env. They ship with
simple_wbc_code, or come from NVIDIA's GR00T whole-body-control release.

What needs the private ones
---------------------------
Driving the scene with the DEPLOYED controller over DDS, and running the VLA
client against it, need vla_for_simulation_testing (gear_sonic for the DDS
bridge, deploy-wbc-on-robot for the container). If you cannot access that
repository, ask whoever owns your control stack for it; there is no public
substitute, and this repo deliberately does not ship a copy.

Weights are never in git
------------------------
VLA checkpoints are tens of GB and are not submodules. Fetch them separately
and set RS_GROOT_CHECKPOINT in configs/paths.env.
EOF
