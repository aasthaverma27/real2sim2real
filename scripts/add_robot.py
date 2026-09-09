#!/usr/bin/env python3
"""Attach a real robot MJCF into a RoboSnap-generated MuJoCo scene.

The reconstructed humanoid is a single rigid mesh in a settled pose - it has no
joints and cannot be actuated. The manufacturer's MJCF brings real kinematics,
joint limits and actuators. This merges the two with MuJoCo's MjSpec API, which
namespaces the robot and resolves its mesh paths independently of the scene's
(a plain <include> fails: MJCF permits only one <compiler> block).

    python scripts/add_robot.py --scene-dir outputs/video_scene \
        --robot assets/robots/g1/g1_29dof_with_hand.xml \
        --at 0.149 1.392
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np


def foot_offset(robot_xml: Path) -> float:
    """Root-to-lowest-point distance at the model's default pose.

    Uses actual mesh vertices; geom_rbound is a bounding *sphere* radius and
    overestimates badly for limb meshes, which leaves the robot hovering.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root_z = d.xpos[1][2]
    low = float("inf")
    for gi in range(m.ngeom):
        pos, mat = d.geom_xpos[gi], d.geom_xmat[gi].reshape(3, 3)
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[gi])
            a = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
            v = m.mesh_vert[a:a + n].reshape(-1, 3)
            low = min(low, float((v @ mat.T + pos)[:, 2].min()))
        else:
            low = min(low, float(pos[2] - m.geom_size[gi].max()))
    return float(root_z - low)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    # The G1 lives in the repo under assets/robots/g1 so it can be edited here
    # instead of in whichever modinoid_sim checkout happens to be on the box -
    # there are five copies on this machine and they are not interchangeable.
    p.add_argument("--robot", type=Path,
                   default=Path(__file__).resolve().parent.parent
                           / "assets/robots/g1/g1_29dof_with_hand.xml",
                   help="Robot MJCF (default: the vendored G1 29-DoF with hands)")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"), required=True,
                   help="Ground position for the robot in scene coordinates.")
    p.add_argument("--floor-z", type=float, default=None,
                   help="Floor height. Default: read from the scene's floor plane.")
    p.add_argument("--yaw", type=float, default=0.0, help="Rotation about Z, radians.")
    p.add_argument("--prefix", default="g1_")
    p.add_argument("--output", type=Path)
    p.add_argument("--fix-base", action="store_true",
                   help="Weld the robot's root to the world instead of leaving "
                        "it on a freejoint. For table-top manipulation this "
                        "mirrors the hardware: arm_sdk lends you the arms and "
                        "waist while the on-board controller keeps the legs "
                        "balancing, so the base is not ours to move. Without "
                        "it, a scene with no balance policy has the robot lean "
                        "forward into the table within a couple of seconds.")
    p.add_argument("--strip-rangefinders", action="store_true",
                   help="Delete the MJCF's rangefinder sensors. The G1 carries "
                        "72 of them; they render as a yellow fan and were "
                        "measured at a 206x physics slowdown in batched runs.")
    p.add_argument("--scene-xml", type=Path, default=None,
                   help="Scene MJCF to merge into. Default: "
                        "<scene-dir>/mujoco/scene.xml. Point this at a "
                        "rescaled scene when the raw export is not to scale.")
    a = p.parse_args()

    scene_xml = a.scene_xml or (a.scene_dir / "mujoco" / "scene.xml")
    if not scene_xml.is_file():
        raise SystemExit(f"missing {scene_xml}; run export_mujoco.py first")

    scene = mujoco.MjSpec.from_file(str(scene_xml))
    robot = mujoco.MjSpec.from_file(str(a.robot))

    floor_z = a.floor_z
    if floor_z is None:
        m = scene.compile()
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        floor_z = float(m.geom_pos[gid][2]) if gid >= 0 else 0.0

    dz = foot_offset(a.robot)
    z = floor_z + dz
    print(f"[add-robot] floor z={floor_z:+.3f}, pelvis offset={dz:.3f} -> pelvis z={z:+.3f}")

    # The robot MJCF usually bakes a scene position into its root body
    # (e.g. <body name="pelvis" pos="3.5 2 1.0">). A frame ADDS to that, so zero
    # it first and let the frame do the placing.
    root = robot.worldbody.bodies[0]
    print(f"[add-robot] robot root '{root.name}' baked pos {list(root.pos)} -> zeroed")
    root.pos = [0.0, 0.0, 0.0]

    if a.fix_base:
        # MjSpec has no weld-on-attach, but deleting the root's freejoint has
        # the same effect: the body keeps the frame it is attached at and gains
        # no dofs. MjsJoint exposes no .delete(), so it goes through the spec.
        freed = [j for j in root.joints
                 if j.type == mujoco.mjtJoint.mjJNT_FREE]
        for j in freed:
            print(f"[add-robot] removing root freejoint '{j.name}' (--fix-base)")
            robot.delete(j)
        if not freed:
            print("[add-robot] --fix-base: root had no freejoint already")

    if a.strip_rangefinders:
        n = 0
        for sensor in list(robot.sensors):
            if sensor.type == mujoco.mjtSensor.mjSENS_RANGEFINDER:
                robot.delete(sensor)
                n += 1
        print(f"[add-robot] stripped {n} rangefinder sensors")

    frame = scene.worldbody.add_frame()
    frame.pos = [a.at[0], a.at[1], z]
    frame.quat = [np.cos(a.yaw / 2), 0, 0, np.sin(a.yaw / 2)]
    scene.attach(robot, prefix=a.prefix, frame=frame)

    m = scene.compile()
    print(f"[add-robot] merged: nbody={m.nbody} nu={m.nu} (actuators) nq={m.nq}")

    # The merged model has a single meshdir (the scene's), so the robot's mesh
    # files must live there too; otherwise MuJoCo looks for assets/pelvis.STL.
    assets = a.scene_dir / "mujoco" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    robot_root = a.robot.resolve().parent
    rspec = mujoco.MjSpec.from_file(str(a.robot))
    rmeshdir = (robot_root / (rspec.meshdir or "")).resolve()
    copied = 0
    for src in list(rmeshdir.rglob("*.STL")) + list(rmeshdir.rglob("*.stl")) \
             + list(rmeshdir.rglob("*.obj")) + list(rmeshdir.rglob("*.msh")):
        dst = assets / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"[add-robot] copied {copied} robot mesh files into {assets}")

    out = a.output or (a.scene_dir / "mujoco" / "scene_with_robot.xml")
    out.write_text(scene.to_xml(), encoding="utf-8")
    print(f"[add-robot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
