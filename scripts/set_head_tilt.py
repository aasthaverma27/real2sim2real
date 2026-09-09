#!/usr/bin/env python3
"""Set the downward pitch of the G1's head cameras, in degrees.

The MJCF hardcodes the head cameras as xyaxes="0 -1 0 0.707 0 0.707", which is
45 degrees down, with no hint that the number is a choice. It is: the ELP
module's tilt is a physical mounting angle on the real robot, and the sim has to
match it or the ego view frames a different part of the scene.

For a downward pitch of theta, with the camera's x axis along body -y:

    forward = ( cos t, 0, -sin t)      x_cam = (0, -1, 0)
    z_cam   = (-cos t, 0,  sin t)      y_cam = (sin t, 0, cos t)

so xyaxes is "0 -1 0  sin(t) 0 cos(t)". Works on either the robot MJCF (affects
every scene built afterwards) or an already-built scene (affects that scene now,
no rebuild).

    scripts/set_head_tilt.py --xml outputs/g1_ok/mujoco/scene_g1.xml --pitch 15
    scripts/set_head_tilt.py --xml assets/robots/g1/g1_29dof_with_hand.xml --pitch 15
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

CAMS = ("elp_left", "elp_right", "head_pinhole")


def quat_for_pitch(deg: float) -> np.ndarray:
    import mujoco
    t = np.deg2rad(deg)
    x = np.array([0.0, -1.0, 0.0])
    y = np.array([np.sin(t), 0.0, np.cos(t)])
    z = np.cross(x, y)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).reshape(9))
    return q


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--pitch", type=float, required=True,
                   help="degrees below horizontal; 0 looks straight ahead")
    p.add_argument("--prefix", default="", help="'g1_' for a built scene")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    q = quat_for_pitch(a.pitch)
    t = np.deg2rad(a.pitch)
    s = a.xml.read_text()
    n = 0
    for cam in CAMS:
        name = a.prefix + cam
        m = re.search(rf'<camera\s+name="{re.escape(name)}"(.*?)/>', s, re.S)
        if not m:
            print(f"  {name}: not in this file, skipped")
            continue
        block = m.group(0)
        new = re.sub(r'\s*(quat|xyaxes)="[^"]*"', "", block)
        new = new.replace(
            f'name="{name}"',
            f'name="{name}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')
        s = s[:m.start()] + new + s[m.end():]
        print(f"  {name}: pitch -> {a.pitch:.1f} deg down")
        n += 1
    if not n:
        raise SystemExit("no head cameras matched; wrong --prefix?")

    # what the tilt does to a camera sitting 10 cm forward of the neck pivot,
    # which is the measurement that is easy to take on the real robot
    print(f"\na camera 10.0 cm forward of the neck pivot drops "
          f"{10 * np.sin(t):.1f} cm and reaches {10 * np.cos(t):.1f} cm forward "
          f"at {a.pitch:.1f} deg")
    if a.dry_run:
        print("dry run, nothing written")
        return 0
    a.xml.write_text(s)
    print(f"wrote {a.xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
