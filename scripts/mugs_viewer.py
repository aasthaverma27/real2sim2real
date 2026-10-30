#!/usr/bin/env python3
"""Live interactive viewer driven by MuGS's own GaussianSensor.

This drives `mugs.sensors.GaussianSensor.render`
once per frame, so what is on screen comes out of the same code path as
`mugs_render.py` and any MuGS rollout: MuGS's segmentation mask, its depth test,
its gsplat call. Nothing here reimplements the compositing.

The sensor renders its MuJoCo layer from a *named* camera, so orbiting means
moving that camera in the model rather than keeping a viewer-side pose: the
world camera's pos/quat are written each frame and the same resulting pose is
handed to the splat side, which is what keeps the two layers locked together.

Mouse: RIGHT-drag orbit, WHEEL zoom, LEFT-drag an object
Keys:  SPACE run physics, B cycle hybrid/3dgs/mujoco, C snap to the capture
       pose, R reset, F re-frame, ESC quit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera quat (wxyz) looking from eye at target: -z forward, +y up."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    zc = -fwd
    up = np.array([0.0, 0.0, 1.0])
    if abs(zc @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    xc = np.cross(up, zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.column_stack([xc, yc, zc])
    q = np.empty(4)
    import mujoco
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return q


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, default=None)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--camera", default="scene_view", help="world camera to drive")
    p.add_argument("--capture-camera", default="cap",
                   help="calibrated frame-0 camera; C snaps back to it")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--fovy", type=float, default=None)
    p.add_argument("--exclude-geoms", default="floor,ground,room")
    p.add_argument("--splat-geoms", default="",
                   help="comma-separated substrings; these geoms keep the mesh's\n                        silhouette but are painted with the splat's colour, the\n                        way Re3Sim substitutes its table and ground pixels")
    p.add_argument("--look-at", type=float, nargs=3, default=None)
    p.add_argument("--distance", type=float, default=None)
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--pos", type=int, nargs=2, default=None,
                   help="place the window here, so several can sit side by side")
    p.add_argument("--mirror-frame", choices=["auto", "identity"], default="auto",
                   help="'auto' undoes build_wbc_scene's re-origin when "
                        "scene_wbc.frame.json exists; 'identity' when the "
                        "publisher renders the very same scene file.")
    p.add_argument("--mirror", type=int, nargs="?", const=5601, metavar="PORT",
                   help="take the robot's pose from the running DDS sim instead "
                   "of stepping physics here. Needs scripts/wbc_state_relay.py "
                   "in .venv_sim: that env has unitree_sdk2py and no gsplat, "
                   "this one has gsplat and no unitree_sdk2py, so the state "
                   "crosses as plain UDP.")
    p.add_argument("--prefix", default="g1_", help="robot namespace, with --mirror")
    p.add_argument("--no-mirror-objects", dest="mirror_objects", action="store_false",
                   help="follow only the robot, leaving the scene objects where "
                        "this scene spawned them. For placement work, where the "
                        "sim overwriting a dragged object is a nuisance.")
    p.add_argument("--fixed-camera", default=None,
                   help="render from this camera as it is (e.g. a robot eye); "
                        "orbit is disabled and the camera is not moved")
    p.add_argument("--visual-dr", nargs="?", const="", default=None,
                   metavar="CONFIG",
                   help="apply the sim's visual-randomization seed (with "
                        "--mirror); CONFIG defaults to "
                        "configs/visual_randomization.yaml")
    a = p.parse_args()

    import cv2
    import mujoco
    from mugs.sensors import GaussianSensor, GaussianSensorConfig

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    mirror = None
    if getattr(a, "mirror", None):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from wbc_state_relay import MirrorReceiver
        mirror = MirrorReceiver(model, a.scene_dir, a.prefix, a.mirror, a.mirror_frame,
                                objects=getattr(a, "mirror_objects", True))
        print(f"[viewer] mirroring the DDS sim on udp {a.mirror}", flush=True)

    mujoco.mj_forward(model, data)

    cam_name = a.fixed_camera or a.camera
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise SystemExit(f"no camera named {cam_name!r} in the model")
    if a.fixed_camera is None and int(model.cam_bodyid[cid]) != 0:
        raise SystemExit(f"--camera {a.camera!r} must be on the worldbody")
    cap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.capture_camera)

    if a.fixed_camera:
        # a robot eye has its own optics; leave them alone
        fovy = a.fovy if a.fovy else float(model.cam_fovy[cid])
    else:
        fovy = a.fovy if a.fovy else (float(model.cam_fovy[cap_id]) if cap_id >= 0
                                      else float(model.cam_fovy[cid]))
    model.cam_fovy[cid] = fovy

    exclude = {e.strip() for e in a.exclude_geoms.split(",") if e.strip()}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mugs_render import hide_excluded_objects
    hide_excluded_objects(model, exclude)          # no shadows from left-out objects

    def is_fg(g):
        if int(model.geom_bodyid[g]) == 0:
            return False
        if int(model.geom_group[g]) == 3:      # collision-only
            return False
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        return not any(e in n for e in exclude)

    fg_ids = [g for g in range(model.ngeom) if is_fg(g)]

    splat_pat = {x.strip() for x in a.splat_geoms.split(",") if x.strip()}
    splat_ids = [g for g in fg_ids
                 if any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                        for x in splat_pat)] if splat_pat else None

    modes = ["hybrid", "3dgs_only", "mujoco_only"]
    mode_i = 0

    def make_sensor(mode):
        return GaussianSensor(GaussianSensorConfig(
            width=a.width, height=a.height, background_ply_path=a.ply,
            render_mode=mode, robot_geom_ids=fg_ids,
            splat_geom_ids=splat_ids, use_sh=True,
            depth_composite=True, camera_convention="mujoco"))

    sensor = make_sensor(modes[mode_i])
    visual = None
    if a.visual_dr is not None and mirror is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "realsim"))
        from visual_randomizer import VisualDRFollower
        visual = VisualDRFollower(model, a.visual_dr or None, [sensor],
                                  tag="hybrid")
    print(f"[mugs-viewer] {len(fg_ids)}/{model.ngeom} foreground geoms, "
          f"fovy {fovy:.2f}, camera {cam_name!r}"
          f"{' (fixed)' if a.fixed_camera else ''}", flush=True)

    def frame_scene():
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in fg_ids:
            pos = data.geom_xpos[g]
            r = float(model.geom_rbound[g]) or float(model.geom_size[g].max())
            lo = np.minimum(lo, pos - r); hi = np.maximum(hi, pos + r)
        if not np.all(np.isfinite(lo)):
            return np.array([0.0, 0.0, 0.5]), 2.0
        return (lo + hi) / 2, max(float(np.linalg.norm(hi - lo) / 2) * 1.9, 0.5)

    auto_t, dist0 = frame_scene()

    # The auto-framing target is derived from the bodies, so any time the scene
    # is nudged the opening viewpoint drifts with it. --look-at/--azimuth/... pin
    # the camera to the world instead, which is what lets a fit change move the
    # objects while the view stays exactly where it was.
    target = np.asarray(a.look_at, float) if a.look_at else auto_t
    if cap_id >= 0:
        eye0 = model.cam_pos[cap_id].copy()
    else:
        eye0 = target + np.array([0.0, -dist0, dist0 * 0.4])
    d = eye0 - target
    nd = max(float(np.linalg.norm(d)), 1e-9)
    cs = {"dist": a.distance if a.distance else nd,
          "az": a.azimuth if a.azimuth is not None
                else float(np.degrees(np.arctan2(d[1], d[0]))),
          "el": a.elevation if a.elevation is not None
                else float(np.degrees(np.arcsin(d[2] / nd)))}
    home = dict(cs); home_t = target.copy()

    ui = {"orbit": False, "last": None, "sel": -1, "drag": False, "dlast": None}
    state = {"seg": None, "R": np.eye(3)}

    def free_joint_of(bid):
        while bid > 0:
            for j in range(model.njnt):
                if int(model.jnt_bodyid[j]) == bid and \
                   int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                    return int(model.jnt_qposadr[j])
            bid = int(model.body_parentid[bid])
        return None

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_RBUTTONDOWN:
            ui["orbit"], ui["last"] = True, (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            ui["orbit"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            cs["dist"] = float(np.clip(
                cs["dist"] * (1.12 if (flags >> 16) < 0 else 1 / 1.12), 0.15, 60.0))
        elif event == cv2.EVENT_LBUTTONDOWN and state["seg"] is not None:
            gid = int(state["seg"][min(max(y, 0), a.height - 1),
                                   min(max(x, 0), a.width - 1)])
            if gid >= 0:
                adr = free_joint_of(int(model.geom_bodyid[gid]))
                if adr is not None:
                    ui["sel"], ui["drag"], ui["dlast"] = adr, True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            ui["drag"], ui["sel"] = False, -1
        elif event == cv2.EVENT_MOUSEMOVE:
            if ui["orbit"] and ui["last"]:
                ox, oy = ui["last"]; ui["last"] = (x, y)
                cs["az"] += (x - ox) * 0.4
                cs["el"] = float(np.clip(cs["el"] - (y - oy) * 0.4, -89, 89))
            elif ui["drag"] and ui["sel"] >= 0 and ui["dlast"]:
                px, py = ui["dlast"]; ui["dlast"] = (x, y)
                sc = 2.0 * cs["dist"] * np.tan(np.deg2rad(fovy) / 2) / a.height
                R = state["R"]           # camera x right, y up in world
                data.qpos[ui["sel"]:ui["sel"] + 3] += (R[:, 0] * (x - px) * sc
                                                       - R[:, 1] * (y - py) * sc)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)

    win = (f"robot camera: {a.fixed_camera}" if a.fixed_camera
           else "MuGS hybrid - GaussianSensor")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.resizeWindow(win, a.width, a.height)
    if a.pos:
        cv2.moveWindow(win, a.pos[0], a.pos[1])

    # Freeze the robot's floating base while physics runs.
    #
    # Holding the actuators at their rest angles is NOT enough: measured, the
    # pelvis still drops 0.792 -> 0.189 m in two seconds, identical to leaving
    # ctrl at zero. Position targets cannot balance a humanoid - that needs the
    # FixStand gains in scripts/apple_scene_demo.py. This is a viewer, so the
    # base is pinned kinematically instead: the robot stays upright and the
    # objects on the table still fall and settle normally.
    base_adr = None
    for j in range(model.njnt):
        b = int(model.jnt_bodyid[j])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE and "pelvis" in nm:
            base_adr = int(model.jnt_qposadr[j])
            base_dof = int(model.jnt_dofadr[j])
            break
    base_q = data.qpos[base_adr:base_adr + 7].copy() if base_adr is not None else None
    if base_q is not None:
        print(f"[mugs-viewer] pinning the robot's floating base while stepping "
              f"(actuator holds alone do not keep it upright)", flush=True)

    hold = None
    if model.nu:
        hold = np.zeros(model.nu)
        for i in range(model.nu):
            j = int(model.actuator_trnid[i, 0])
            if j >= 0 and int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE,
                                                     mujoco.mjtJoint.mjJNT_SLIDE):
                hold[i] = data.qpos[int(model.jnt_qposadr[j])]

    running = False
    while True:
        if running:
            for _ in range(10):
                if hold is not None:
                    data.ctrl[:] = hold
                if mirror is not None:
                    mirror.apply(data)          # the DDS sim owns the physics
                    if visual is not None:
                        visual.update(mirror)
                else:
                    mujoco.mj_step(model, data)
                if base_q is not None:
                    data.qpos[base_adr:base_adr + 7] = base_q
                    data.qvel[base_dof:base_dof + 6] = 0.0

        if not a.fixed_camera:
            ar, er = np.deg2rad(cs["az"]), np.deg2rad(cs["el"])
            eye = target + cs["dist"] * np.array(
                [np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
            model.cam_pos[cid] = eye
            model.cam_quat[cid] = look_at_quat(eye, target)
        mujoco.mj_forward(model, data)

        R = data.cam_xmat[cid].reshape(3, 3).copy()
        state["R"] = R
        f_px = (a.height / 2) / np.tan(np.deg2rad(fovy) / 2)
        out = sensor.render(model, data, cam_name, return_components=True,
                            camera_params={"position": data.cam_xpos[cid].copy(),
                                           "rotation": R, "fx": f_px, "fy": f_px,
                                           "width": a.width, "height": a.height})
        if isinstance(out, dict):
            rgb, state["seg"] = out["rgb"], out.get("seg_ids")
            vis = 100.0 * out["visible"].astype(bool).mean()
        else:
            rgb, vis = out, float("nan")

        img = np.ascontiguousarray(rgb[:, :, ::-1])
        hud = (f"{cam_name}  fovy {fovy:.0f}" if a.fixed_camera
               else f"MuGS {modes[mode_i]}  d={cs['dist']:.2f}m")
        cv2.putText(img, f"{hud}  {'RUN' if running else 'PAUSE'}  fg={vis:.0f}%",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.imshow(win, img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            running = not running
        elif k in (ord('r'), ord('R')):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            running = False
        elif k in (ord('c'), ord('C')):
            cs.update(home); target = home_t.copy()
        elif k in (ord('f'), ord('F')):
            if a.look_at:
                target = home_t.copy(); cs.update(home)
            else:
                target, cs["dist"] = frame_scene()
        elif k in (ord('b'), ord('B')):
            mode_i = (mode_i + 1) % len(modes)
            sensor.close(); sensor = make_sensor(modes[mode_i])
            if visual is not None:
                visual.sensors.clear(); visual.attach(sensor)
            print(f"[mugs-viewer] mode -> {modes[mode_i]}", flush=True)
        elif k == 81: cs["az"] -= 3
        elif k == 83: cs["az"] += 3
        elif k == 82: cs["el"] = min(cs["el"] + 3, 89)
        elif k == 84: cs["el"] = max(cs["el"] - 3, -89)
        elif k in (ord('='), ord('+')): cs["dist"] = max(0.15, cs["dist"] / 1.12)
        elif k in (ord('-'), ord('_')): cs["dist"] = min(60.0, cs["dist"] * 1.12)

    sensor.close(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
