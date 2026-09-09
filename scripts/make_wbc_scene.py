#!/usr/bin/env python3
"""Build a gear_sonic-compatible MJCF: their G1 standing in a RoboSnap scene.

The GEAR WBC stack is left completely alone - the container, the ONNX policies
and the DDS control loop all run as-is. The only thing that changes is the scene
their MuJoCo sim loads (ROBOT_SCENE in the sim's wbc yaml).

That sim is strict about the robot, which is why the merge cannot reuse
outputs/<scene>/mujoco/scene_with_robot.xml:

  * bodies must be named `pelvis` / `torso_link` and the base joint
    `floating_base_joint` - add_robot.py namespaces them to `g1_*`;
  * joint state is read as qpos[joint_id + 6], so the robot's free joint must be
    joint 0 and its 43 hinges joints 1..43, with nothing in front of them.

So the robot MJCF is the parent here and the RoboSnap scene is attached into it
(the mirror of add_robot.py), prefixed and transformed so that the spot where
the robot stands in the reconstruction lands on the sim's origin.

    python scripts/make_wbc_scene.py --scene-dir outputs/video_scene
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np

GEAR_G1 = Path(os.environ.get(
    "RS_DEPLOY_ROOT", "/mnt/ssd2/vla_for_simulation_testing")
) / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
ASSET_SUFFIXES = ("*.STL", "*.stl", "*.obj", "*.msh", "*.png", "*.jpg", "*.mtl")


def robot_pose(scene_with_robot: Path, prefix: str) -> tuple[np.ndarray, float]:
    """Where add_robot.py put the robot: ground position and yaw, scene frame."""
    spec = mujoco.MjSpec.from_file(str(scene_with_robot))
    model = spec.compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")
    if bid < 0:
        raise SystemExit(f"no body '{prefix}pelvis' in {scene_with_robot}")
    pos = np.array(model.body_pos[bid], dtype=float)
    w, x, y, z = model.body_quat[bid]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pos, yaw


def floor_height(scene: Path) -> float:
    model = mujoco.MjSpec.from_file(str(scene)).compile()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return float(model.geom_pos[gid][2]) if gid >= 0 else 0.0


def copy_assets(spec_file: Path, dst: Path) -> int:
    """Copy every mesh and texture a spec references into one flat directory.

    The merged model has a single meshdir, so both the robot's STLs and the
    reconstruction's OBJs have to sit in it.
    """
    spec = mujoco.MjSpec.from_file(str(spec_file))
    root = spec_file.resolve().parent
    n = 0
    for sub in {spec.meshdir or "", spec.texturedir or ""}:
        for pattern in ASSET_SUFFIXES:
            for src in (root / sub).resolve().rglob(pattern):
                out = dst / src.name
                if not out.exists():
                    shutil.copy2(src, out)
                    n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--robot", type=Path, default=GEAR_G1,
                   help="the MJCF gear_sonic's sim expects (default: their g1_29dof_with_hand)")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                   help="robot's ground position in scene coords. "
                        "Default: read it back from scene_with_robot.xml.")
    p.add_argument("--yaw", type=float, help="robot heading, radians. Default: same source.")
    p.add_argument("--prefix", default="rs_", help="namespace for the scene's own bodies")
    p.add_argument("--output", type=Path,
                   help="default: <scene>/mujoco/wbc/scene_wbc.xml")
    a = p.parse_args()

    scene_xml = a.scene_dir / "mujoco" / "scene.xml"
    if not scene_xml.is_file():
        raise SystemExit(f"missing {scene_xml}; run export_mujoco.py first")
    if not a.robot.is_file():
        raise SystemExit(f"missing {a.robot}")

    at, yaw = a.at, a.yaw
    if at is None or yaw is None:
        placed = a.scene_dir / "mujoco" / "scene_with_robot.xml"
        if not placed.is_file():
            raise SystemExit(f"pass --at/--yaw, or run add_robot.py to create {placed}")
        pos, y0 = robot_pose(placed, "g1_")
        at = at or pos[:2]
        yaw = y0 if yaw is None else yaw
    floor_z = floor_height(scene_xml)
    print(f"[wbc-scene] robot stands at ({at[0]:+.3f}, {at[1]:+.3f}) yaw {np.degrees(yaw):+.1f}deg, "
          f"floor z={floor_z:+.3f}")

    robot = mujoco.MjSpec.from_file(str(a.robot))
    scene = mujoco.MjSpec.from_file(str(scene_xml))

    # Map the reconstruction into the sim's frame: undo the robot's placement so
    # its feet land on the origin, and lift the floor to z=0. A frame applies
    # p_world = R_f @ p_local + p_f, so R_f is the inverse yaw and p_f cancels
    # the translation that survives it.
    rot = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
                    [np.sin(-yaw), np.cos(-yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    origin = np.array([at[0], at[1], floor_z], dtype=float)
    frame = robot.worldbody.add_frame()
    frame.pos = (-rot @ origin).tolist()
    frame.quat = [np.cos(-yaw / 2), 0.0, 0.0, np.sin(-yaw / 2)]
    robot.attach(scene, prefix=a.prefix, frame=frame)

    # The robot MJCF is only the robot: no light, no sky. Their scene_43dof.xml
    # adds those, and without them the viewer renders a black void.
    light = robot.worldbody.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    out = a.output or (a.scene_dir / "mujoco" / "wbc" / "scene_wbc.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = (out.parent / "assets").resolve()
    assets.mkdir(exist_ok=True)
    n = copy_assets(a.robot, assets) + copy_assets(scene_xml, assets)
    print(f"[wbc-scene] copied {n} mesh files into {assets}")

    # Absolute, because the merged model keeps one meshdir and it is resolved
    # against the robot MJCF's directory, not the file we are about to write.
    robot.meshdir = str(assets)
    robot.texturedir = str(assets)

    model = robot.compile()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if base != 0:
        raise SystemExit("floating_base_joint is not joint 0 - the sim's qpos indexing "
                         "would read the wrong joints")
    hinges = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 44)]
    stray = [n for n in hinges if n and n.startswith(a.prefix)]
    if stray:
        raise SystemExit(f"scene joints landed inside the robot's block: {stray[:3]}")
    print(f"[wbc-scene] merged: nbody={model.nbody} nu={model.nu} nq={model.nq}, "
          f"robot occupies joints 0..43")

    out.write_text(robot.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(out))      # fails loudly if meshes are missing
    print(f"[wbc-scene] wrote {out}")
    print(f"\nPoint the sim at it:\n  ROBOT_SCENE: \"{out.resolve()}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
