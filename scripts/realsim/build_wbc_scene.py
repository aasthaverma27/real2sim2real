#!/usr/bin/env python3
"""Merge a reconstructed scene into a robot MJCF, the way the reference sim needs.

This is the step that makes a RoboSnap scene drivable by a real whole-body
controller, and the direction of the merge is the whole point.

`add_robot.py` attaches the ROBOT into the SCENE. That is right for rendering,
but a deployed WBC sim is strict about the robot and such a scene breaks it:

  * bodies must keep their own names (`pelvis`, `torso_link`) and the base joint
    must be `floating_base_joint` - attaching into the scene namespaces them;
  * joint state is read as `qpos[joint_id + 6]`, so the robot's free joint has
    to be joint 0 and its hinges 1..N with nothing in front of them.

So here the ROBOT is the parent and the SCENE is attached into it, prefixed and
transformed so that the spot where the robot stands in the reconstruction lands
on the sim's origin. Nothing about the controller, its ONNX policies or its
control loop changes - only the scene it loads.

Robot-agnostic: every robot-specific value comes from the descriptor
(configs/robots/*.yaml), so a different robot is a different --robot, not a
different script.

    scripts/realsim/build_wbc_scene.py --scene-dir outputs/g1_ok \
        --robot g1_29dof_with_hand --at -0.02 1.37 --yaw -1.62
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robosnap.realsim import load_robot

ASSET_SUFFIXES = ("*.STL", "*.stl", "*.obj", "*.msh", "*.png", "*.jpg", "*.mtl")


def placed_pose(scene_with_robot: Path, root_body: str,
                prefix: str) -> tuple[np.ndarray, float]:
    """Read back where add_robot.py put the robot: ground position and yaw."""
    model = mujoco.MjSpec.from_file(str(scene_with_robot)).compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{root_body}")
    if bid < 0:
        raise SystemExit(f"no body '{prefix}{root_body}' in {scene_with_robot.name}")
    pos = np.array(model.body_pos[bid], dtype=float)
    w, x, y, z = model.body_quat[bid]
    return pos, float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def strip_robot(scene: "mujoco.MjSpec", prefix: str) -> int:
    """Remove an already-attached robot from a scene spec, in place.

    Deleting a body takes its whole subtree, so only the roots need naming.
    This lets the merge use the scene you are ACTUALLY driving - the one whose
    object placements and dimensions were tuned - rather than the untouched
    export, while still handing the robot MJCF the parent role.
    """
    roots = [b for b in scene.bodies
             if b.name.startswith(prefix)
             and not (b.parent is not None and b.parent.name.startswith(prefix))]
    for b in roots:
        scene.delete(b)
    return len(roots)


def floor_height(scene: "mujoco.MjSpec") -> float:
    model = scene.compile()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return float(model.geom_pos[gid][2]) if gid >= 0 else 0.0


def copy_assets(spec_file: Path, dst: Path) -> int:
    """Flatten every mesh and texture a spec references into one directory.

    The merged model keeps a single meshdir, so the robot's STLs and the
    reconstruction's OBJs have to share it.
    """
    spec = mujoco.MjSpec.from_file(str(spec_file))
    root = spec_file.resolve().parent
    n = 0
    for sub in {spec.meshdir or "", spec.texturedir or ""}:
        base = (root / sub).resolve()
        if not base.is_dir():
            continue
        for pattern in ASSET_SUFFIXES:
            for src in base.rglob(pattern):
                out = dst / src.name
                if not out.exists():
                    shutil.copy2(src, out)
                    n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--robot", default="g1_29dof_with_hand",
                   help="descriptor name in configs/robots/, or a path to one")
    p.add_argument("--mjcf", type=Path, default=None,
                   help="override the descriptor's robot MJCF (e.g. to use the "
                        "vendor copy the deployed sim already trusts)")
    p.add_argument("--scene-xml", type=Path, default=None,
                   help="default: <scene-dir>/mujoco/scene.xml")
    p.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                   help="robot's ground position in scene coords. Default: read "
                        "it back from scene_with_robot.xml if that exists.")
    p.add_argument("--yaw", type=float, help="robot heading, radians")
    p.add_argument("--placed-prefix", default="g1_",
                   help="the namespace add_robot.py used, when reading --at back")
    p.add_argument("--strip-prefix", default=None, metavar="PREFIX",
                   help="remove an already-attached robot from --scene-xml "
                        "before merging (e.g. g1_). Use this to merge the scene "
                        "you are actually driving instead of the raw export.")
    p.add_argument("--prefix", default="rs_",
                   help="namespace for the SCENE's bodies inside the merged model")
    p.add_argument("--offscreen", type=int, nargs=2, default=[1920, 1920],
                   metavar=("W", "H"),
                   help="offscreen framebuffer for the merged model. A robot "
                        "MJCF usually ships a small one, and every offscreen "
                        "renderer (MuGS, the fisheye cubemap) then fails with "
                        "'Image height > framebuffer height'.")
    p.add_argument("--output", type=Path,
                   help="default: <scene-dir>/mujoco/wbc/scene_wbc.xml")
    a = p.parse_args()

    spec = load_robot(a.robot)
    robot_xml = a.mjcf or spec.mjcf
    scene_xml = a.scene_xml or (a.scene_dir / "mujoco" / "scene.xml")
    if not scene_xml.is_file():
        raise SystemExit(f"missing {scene_xml} - run export_mujoco.py first")
    if not Path(robot_xml).is_file():
        raise SystemExit(f"missing robot MJCF {robot_xml}")

    at, yaw = a.at, a.yaw
    if at is None or yaw is None:
        placed = a.scene_dir / "mujoco" / "scene_with_robot.xml"
        if not placed.is_file():
            raise SystemExit(
                f"pass --at X Y --yaw RAD, or run add_robot.py first to create "
                f"{placed} for this script to read the placement back from")
        pos, y0 = placed_pose(placed, spec.root_body, a.placed_prefix)
        at = at if at is not None else pos[:2]
        yaw = y0 if yaw is None else yaw

    print(f"[wbc-scene] robot {spec.name}")
    robot = mujoco.MjSpec.from_file(str(robot_xml))
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    if a.strip_prefix:
        n = strip_robot(scene, a.strip_prefix)
        print(f"[wbc-scene] stripped {n} '{a.strip_prefix}*' root body(ies) "
              f"from {scene_xml.name}")

    floor_z = floor_height(scene)

    print(f"[wbc-scene] stands at ({at[0]:+.3f}, {at[1]:+.3f}) "
          f"yaw {np.degrees(yaw):+.1f} deg, scene floor z={floor_z:+.3f}")

    # Map the reconstruction into the sim's frame: undo the robot's placement so
    # its feet land on the origin, and lift the floor to z=0. A frame applies
    # p_world = R_f @ p_local + p_f, so R_f is the inverse yaw and p_f cancels
    # the translation that survives it.
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([at[0], at[1], floor_z], dtype=float)
    frame = robot.worldbody.add_frame()
    frame.pos = (-rot @ origin).tolist()
    frame.quat = [np.cos(-yaw / 2), 0.0, 0.0, np.sin(-yaw / 2)]
    robot.attach(scene, prefix=a.prefix, frame=frame)

    # A robot MJCF is only the robot - no light, no sky. Without these the
    # viewer renders a black void.
    light = robot.worldbody.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    ow, oh = a.offscreen
    robot.visual.global_.offwidth = max(int(ow), int(robot.visual.global_.offwidth))
    robot.visual.global_.offheight = max(int(oh), int(robot.visual.global_.offheight))

    out = a.output or (a.scene_dir / "mujoco" / "wbc" / "scene_wbc.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = (out.parent / "assets").resolve()
    assets.mkdir(exist_ok=True)
    n = copy_assets(Path(robot_xml), assets) + copy_assets(Path(scene_xml), assets)
    print(f"[wbc-scene] flattened {n} mesh/texture files into {assets}")

    # Absolute, because the merged model keeps one meshdir and MuJoCo resolves
    # it against the robot MJCF's directory, not the file we are about to write.
    robot.meshdir = str(assets)
    robot.texturedir = str(assets)

    model = robot.compile()

    # The two invariants the reference sim depends on. Checked here so a broken
    # scene fails now, with a message, instead of silently driving the wrong
    # joints at 50 Hz later.
    base_name = spec.base_joint
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_name)
    if base < 0:
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 base_name.removesuffix("_joint"))
    if base != 0:
        raise SystemExit(
            f"'{base_name}' compiled to joint {base}, not 0 - the sim indexes "
            f"qpos[joint_id + 6] and would read the wrong joints")
    n_j = spec.n_robot_joints
    if n_j:
        hinges = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                  for i in range(1, n_j + 1)]
        stray = [h for h in hinges if h and h.startswith(a.prefix)]
        if stray:
            raise SystemExit(
                f"scene joints landed inside the robot's block 1..{n_j}: "
                f"{stray[:3]} - the reconstruction must attach after the robot")
        print(f"[wbc-scene] robot occupies joints 0..{n_j}, scene starts after")

    # Record the re-origin so anything rendering in the SCENE frame can undo it.
    # The sim runs this merged model, whose origin is the robot's stance; a
    # viewer on the original scene xml is 'at' metres and 'yaw' radians away.
    # Without this the mirrored robot lands in the wrong place and the hybrid
    # depth test rejects nearly the whole foreground.
    meta = out.with_suffix(".frame.json")
    meta.write_text(json.dumps({
        "at": [float(at[0]), float(at[1])],
        "yaw": float(yaw),
        "floor_z": float(floor_z),
        "scene_xml": str(Path(scene_xml).resolve()),
        "robot": spec.name,
    }, indent=1), encoding="utf-8")
    print(f"[wbc-scene] wrote {meta.name} (the frame shift the viewers undo)")

    out.write_text(robot.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(out))      # fails loudly on missing meshes
    print(f"[wbc-scene] merged: nbody={model.nbody} nq={model.nq} nu={model.nu}")
    print(f"[wbc-scene] wrote {out}")
    print(f"\nPoint the reference sim at it:\n  ROBOT_SCENE: \"{out.resolve()}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
