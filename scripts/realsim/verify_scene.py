#!/usr/bin/env python3
"""Check a built scene against everything that has silently broken before.

Each check here exists because the failure it catches was found the hard way,
and every one of them presents as "the stack runs but the robot behaves oddly"
rather than as an error. Run it after building a scene, or any time the robot
model changes:

    python scripts/realsim/verify_scene.py --scene-dir outputs/myroom

Exit code is the number of failures, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from wbc_state_relay import BODY_JOINTS, HAND_JOINTS  # noqa: E402

# Actuator order of the shipped G1 hands (assets/robots/g1/*.xml): thumb first,
# then the two fingers - and the two hands differ in which finger comes second.
# The sim maps rt/dex3/<side>/cmd motor_cmd[i] onto actuator 29+i, and this is
# the order the deployed controller drives correctly. Reordering the actuators
# to "match" the Dex3 SDK's motor numbering scrambles the fingers: every finger
# then executes another finger's command, with no error anywhere.
HAND_LEFT = ["thumb_0", "thumb_1", "thumb_2",
             "middle_0", "middle_1", "index_0", "index_1"]
HAND_RIGHT = ["thumb_0", "thumb_1", "thumb_2",
              "index_0", "index_1", "middle_0", "middle_1"]

OK, BAD = "  ok  ", " FAIL "


class Report:
    def __init__(self) -> None:
        self.fails = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"[{OK if ok else BAD}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            self.fails += 1
        return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    a = p.parse_args()

    scene = a.scene_dir
    render_xml = scene / "mujoco" / "scene_g1.xml"
    wbc_xml = scene / "mujoco" / "wbc" / "scene_wbc.xml"
    r = Report()

    print(f"\n=== verifying {scene} ===\n")

    if not r.check(render_xml.is_file(), "render scene exists", str(render_xml)):
        return r.fails
    if not r.check(wbc_xml.is_file(), "sim scene exists", str(wbc_xml)):
        return r.fails

    m = mujoco.MjModel.from_xml_path(str(wbc_xml))
    rm = mujoco.MjModel.from_xml_path(str(render_xml))

    # --- what the controller requires of the sim scene --------------------
    j0 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, 0)
    r.check(j0 == "floating_base_joint", "base joint is joint 0", f"got '{j0}'")

    types = [int(m.jnt_type[j]) for j in range(1, 44)]
    r.check(all(t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                for t in types),
            "joints 1..43 are hinges", "the WBC reads qpos[joint_id + 6]")

    # --- the 51-float wire format the photoreal windows read ---------------
    have = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(m.njnt)}
    missing = [n + "_joint" for n in BODY_JOINTS + HAND_JOINTS
               if n + "_joint" not in have]
    r.check(not missing, "all 43 wire joints present", ", ".join(missing[:4]))

    # --- hand actuator order (silent finger scrambling) --------------------
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    for side, order, base in (("left", HAND_LEFT, 29), ("right", HAND_RIGHT, 36)):
        got = [(names[base + i] or "").replace(f"{side}_hand_", "")
               for i in range(7)]
        r.check(got == order, f"{side} hand actuators in thumb-first order",
                f"got {got}")

    # --- finger joint dynamics (violent fingers) ---------------------------
    slack = []
    for n in HAND_JOINTS:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
        if j < 0:
            continue
        d = int(m.jnt_dofadr[j])
        if m.dof_armature[d] <= 0 or m.dof_damping[d] <= 0:
            slack.append(n)
    r.check(not slack, "finger joints have armature and damping",
            f"{len(slack)} without: a bare finger link is ~2e-4 kg m^2 and the "
            f"WBC's torque slams it between its limits")

    # --- objects the sim owns and streams ---------------------------------
    free = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    movable = [f for f in free if f != "floating_base_joint"]
    r.check(True, f"{len(movable)} movable object(s)",
            ", ".join(movable) or "none - the scene has only static geometry")

    # --- the render scene the windows load --------------------------------
    cams = [mujoco.mj_id2name(rm, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(rm.ncam)]
    r.check(any("head" in (c or "") or "ego" in (c or "") for c in cams),
            "render scene has an ego camera", ", ".join(c for c in cams if c))

    plys = sorted(p.name for p in scene.glob("gs_bg*.ply"))
    r.check(bool(plys), "background splat present", ", ".join(plys))

    # --- the rounding trap in frame.json -----------------------------------
    fj = scene / "mujoco" / "wbc" / "scene_wbc.frame.json"
    if fj.is_file():
        f = json.loads(fj.read_text())
        q = rm.body("g1_pelvis").quat if mujoco.mj_name2id(
            rm, mujoco.mjtObj.mjOBJ_BODY, "g1_pelvis") >= 0 else None
        if q is not None:
            actual = float(2 * np.arctan2(q[3], q[0]))
            drift = abs(actual - float(f["yaw"]))
            r.check(drift < 1e-3,
                    "frame.json yaw matches the built scene",
                    f"stored {f['yaw']:.6f} vs actual {actual:.6f} "
                    f"({np.degrees(drift):.3f} deg) - rebuilding from the stored "
                    f"value would move the robot off its own splat")

    print()
    if r.fails:
        print(f"=== {r.fails} check(s) FAILED - see README > Troubleshooting ===\n")
    else:
        print("=== all checks passed ===\n")
    return r.fails


if __name__ == "__main__":
    raise SystemExit(main())
