#!/usr/bin/env bash
# Put the stock Unitree G1 meshes next to the robot MJCFs in assets/robots/g1/.
#
#   bash scripts/fetch_g1_meshes.sh                 # from NVIDIA GR00T-WholeBodyControl
#   bash scripts/fetch_g1_meshes.sh --from <dir>    # from any G1 mesh folder
#
# The robot models belong to Unitree and are not committed here (.gitignore).
# Only the camera-rig parts in assets/robots/g1/meshes/cad/ are ours and ship
# with the repo. The stock meshes - body links plus the three-finger hands -
# come from the gear_sonic G1 model in NVIDIA's GR00T-WholeBodyControl
# (external/, fetched by scripts/setup_external.sh), the same model the
# deployed whole-body controller uses.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/_bootstrap.sh" || exit 1

SRC="$RS_DEPLOY_ROOT/gear_sonic/data/robot_model/model_data/g1/meshes"
[ "${1:-}" = "--from" ] && SRC="$2"
DST="$ROOT/assets/robots/g1/meshes"

[ -d "$SRC" ] || { echo "no mesh folder at $SRC" >&2
                   echo "run scripts/setup_external.sh, or pass --from <dir>" >&2; exit 1; }
mkdir -p "$DST"
cp -n "$SRC"/*.STL "$SRC"/*.stl "$DST"/ 2>/dev/null || true

# every mesh the shipped robots reference must now resolve
"$RS_PY_SIM" - "$ROOT/assets/robots/g1" <<'PY'
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
bad = []
for xml in sorted(d.glob("*.xml")):
    s = xml.read_text()
    md = re.search(r'meshdir="([^"]+)"', s)
    base = d / (md.group(1) if md else "")
    bad += [f"{xml.name}: {f}" for f in sorted(set(re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', s)))
            if not (base / f).exists()]
print("\n".join(["missing meshes:"] + bad) if bad else f"all robot meshes present in {d}/meshes")
sys.exit(1 if bad else 0)
PY
