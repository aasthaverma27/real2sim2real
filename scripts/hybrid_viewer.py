#!/usr/bin/env python3
"""Interactive viewer for the MuJoCo + 3DGS hybrid render.

mujoco.viewer cannot show this: the splat is never part of MuJoCo's scene, it is
composited afterwards in numpy. So we drive the loop ourselves - step physics,
render both layers, blend, display.

Mouse: LEFT-drag an object to move it   RIGHT-drag to orbit   WHEEL to zoom
Keys:  SPACE pause/run   R reset   arrows orbit   +/- zoom   W/S raise/lower
       X flip 180 deg   F re-frame on the scene   C jump to the capture viewpoint
       B cycle background/foreground/composite   ESC quit

The orbit target is computed from the model's own geometry, not hard-coded. An
earlier fixed default (target z = -0.2, from a scene whose floor sat at negative
z) put the camera underground on any scene that had since been rescaled, which
looks exactly like a broken splat rather than a misaimed camera.

    bash scripts/run_hybrid_render.sh --help     # same env is required
    MUJOCO_GL=egl python scripts/hybrid_viewer.py --scene-dir outputs/video_scene
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_splat_render import load_gaussians          # reuse the loader


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--mjcf", type=Path)
    p.add_argument("--ply", type=Path)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--fovy", type=float, default=None)
    p.add_argument("--mirror", type=int, nargs="?", const=5601, metavar="PORT",
                   help="show the gear_sonic sim's robot instead of stepping physics "
                        "here; needs scripts/wbc_state_relay.py running")
    p.add_argument("--prefix", default="g1_", help="robot namespace, with --mirror")
    p.add_argument("--look-at", type=float, nargs=3, default=None, metavar=("X","Y","Z"),
                   help="Orbit target. Default: centre of the scene's own geometry.")
    p.add_argument("--distance", type=float, default=None,
                   help="Initial camera distance. Default: framed on the scene.")
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=-20.0)
    a = p.parse_args()

    import cv2, torch, mujoco
    from gsplat import rasterization

    mjcf = a.mjcf or a.scene_dir / "mujoco" / "scene_with_robot.xml"
    ply = a.ply
    if ply is None:                      # newest gravity-aligned background
        cands = sorted(a.scene_dir.glob("gravity_aligned_background*.ply"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit(f"no gravity_aligned_background*.ply in {a.scene_dir}")
        ply = cands[0]
        print(f"[auto] background: {ply.name}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # real intrinsics -> fovy
    fovy = a.fovy
    cj_path = a.scene_dir / "depth" / "camera.json"
    if fovy is None and cj_path.is_file():
        cj = json.loads(cj_path.read_text())
        K0 = np.asarray(cj["intrinsic_original_pixels"], float)
        ow, oh = cj["original_size_wh"]
        fy = K0[1, 1] * (a.height / float(oh))
        fovy = float(np.rad2deg(2 * np.arctan((a.height / 2) / fy)))
    fovy = fovy or 58.0

    # inject the capture-viewpoint camera so the "C" key has something to use
    cap_T = None
    gj0 = a.scene_dir / "gravity_alignment.json"
    mjcf_load = mjcf
    # scripts/warp_gate.py --emit already writes a "capture" camera into the batch
    # MJCF, and MuJoCo rejects a duplicate camera name outright.
    has_capture = 'name="capture"' in mjcf.read_text(encoding="utf-8")
    if gj0.is_file() and not has_capture:
        import trimesh
        cap_T = np.asarray(json.loads(gj0.read_text())["transform_gravity_from_camera"], float)
        R = cap_T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        M = np.eye(4); M[:3, :3] = R
        q = trimesh.transformations.quaternion_from_matrix(M)
        pos = cap_T[:3, 3]
        cxml = (f'<camera name="capture" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{fovy}"/>')
        mjcf_load = mjcf.with_name("_viewer_" + mjcf.name)
        mjcf_load.write_text(mjcf.read_text().replace("</worldbody>", cxml + "</worldbody>"))

    if cap_T is None and gj0.is_file():
        cap_T = np.asarray(json.loads(gj0.read_text())["transform_gravity_from_camera"],
                           float)

    model = mujoco.MjModel.from_xml_path(str(mjcf_load))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    g = load_gaussians(ply, dev)
    sh_deg = int(np.sqrt(g["sh"].shape[1]) - 1)
    print(f"[viewer] {len(g['means']):,} gaussians, fovy={fovy:.2f}")

    renderer = mujoco.Renderer(model, a.height, a.width)
    vopt = mujoco.MjvOption(); mujoco.mjv_defaultOption(vopt)
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    # foreground geoms = everything except worldbody + collision-only
    keep = [gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) != 0 and int(model.geom_group[gid]) != 3]


    # --- orbit state, framed on the actual scene -------------------------
    def frame_scene():
        """Centre and radius of every visible geom, so the camera starts aimed
        at the scene rather than at a coordinate baked in for another one."""
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for gid in keep:
            pos = data.geom_xpos[gid]
            if int(model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[gid])
                adr, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                v = model.mesh_vert[adr:adr + n]
                w = v @ data.geom_xmat[gid].reshape(3, 3).T + pos
                lo = np.minimum(lo, w.min(0)); hi = np.maximum(hi, w.max(0))
            else:
                r = float(model.geom_size[gid].max())
                lo = np.minimum(lo, pos - r); hi = np.maximum(hi, pos + r)
        if not np.all(np.isfinite(lo)):
            return np.array([0.0, 0.0, 0.5]), 2.5
        c = (lo + hi) / 2
        rad = float(np.linalg.norm(hi - lo) / 2)
        return c, max(rad * 2.1, 0.5)

    auto_target, auto_dist = frame_scene()
    target = np.asarray(a.look_at, float) if a.look_at else auto_target.copy()
    dist = a.distance if a.distance else auto_dist
    el = a.elevation
    if a.azimuth is not None:
        az = a.azimuth
    else:
        # look from the robot's side if there is one, else from -x
        rb = [b for b in range(model.nbody)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                  ).startswith(a.prefix)]
        if rb:
            rp = data.xpos[rb[0]]
            az = float(np.degrees(np.arctan2(target[1] - rp[1], target[0] - rp[0])))
        else:
            az = 0.0
    print(f"[viewer] framed on ({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}) "
          f"at {dist:.2f} m, azimuth {az:+.1f} deg")
    print("[viewer] RIGHT-drag orbit | WHEEL zoom | LEFT-drag an object | "
          "SPACE run | B layers | X flip 180 | F re-frame | ESC quit")
    free_cam = True
    running, mode = False, 0
    fy_px = (a.height / 2) / np.tan(np.deg2rad(fovy) / 2)
    K = torch.tensor([[fy_px, 0, a.width / 2], [0, fy_px, a.height / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)

    # --- direct manipulation: click a body, drag it in the camera plane -----
    ui = {"sel": -1, "drag": False, "last": None, "cam_R": np.eye(3),
          "orbit": False, "olast": None}
    cs = {"az": az, "el": el, "dist": dist}   # mutated by the mouse callback

    def free_joint_of(bid):
        """qpos address of a body's free joint, or None if it is not free."""
        while bid > 0:
            for j in range(model.njnt):
                if int(model.jnt_bodyid[j]) == bid and \
                   int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                    return int(model.jnt_qposadr[j])
            bid = int(model.body_parentid[bid])
        return None

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            gid = int(seg_buf[min(max(y, 0), a.height - 1),
                              min(max(x, 0), a.width - 1)]) if seg_buf is not None else -1
            if gid >= 0:
                bid = int(model.geom_bodyid[gid])
                adr = free_joint_of(bid)
                if adr is not None:
                    ui["sel"], ui["drag"], ui["last"] = adr, True, (x, y)
                    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                    print(f"[grab] {nm}")
                else:
                    print("[grab] that body is static")
        elif event == cv2.EVENT_MOUSEMOVE and ui["drag"] and ui["sel"] >= 0:
            px, py = ui["last"]; ui["last"] = (x, y)
            # screen delta -> world delta in the camera's image plane
            scale = 2.0 * dist_now[0] * np.tan(np.deg2rad(fovy) / 2) / a.height
            R = ui["cam_R"]
            data.qpos[ui["sel"]:ui["sel"] + 3] += (R[:, 0] * (x - px) * scale
                                                   - R[:, 1] * (y - py) * scale)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
        elif event == cv2.EVENT_LBUTTONUP:
            ui["drag"] = False; ui["sel"] = -1
        elif event == cv2.EVENT_RBUTTONDOWN:
            ui["orbit"], ui["olast"] = True, (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            ui["orbit"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags carries the wheel delta in the high 16 bits
            step = 1.12 if (flags >> 16) < 0 else 1 / 1.12
            cs["dist"] = float(np.clip(cs["dist"] * step, 0.15, 60.0))
        if event == cv2.EVENT_MOUSEMOVE and ui["orbit"] and ui["olast"] is not None:
            ox, oy = ui["olast"]; ui["olast"] = (x, y)
            cs["az"] -= (x - ox) * 0.4
            cs["el"] = float(np.clip(cs["el"] + (y - oy) * 0.4, -89, 89))

    # --- mirror mode: the robot's pose comes from the running gear_sonic sim ---
    mirror = None
    if a.mirror:
        from wbc_state_relay import MirrorReceiver
        mirror = MirrorReceiver(model, a.scene_dir, a.prefix, a.mirror)
        print(f"[viewer] mirroring the sim on udp {a.mirror} "
              f"(yaw {np.degrees(mirror.yaw):+.1f}deg, "
              f"origin {np.round(mirror.origin, 3).tolist()})")

    seg_buf = None
    dist_now = [dist]
    cv2.namedWindow("MuJoCo + 3DGS", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MuJoCo + 3DGS", on_mouse)
    cv2.resizeWindow("MuJoCo + 3DGS", a.width, a.height)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)

    while True:
        if mirror is not None:
            # The sim owns the physics; here we only place the robot and redraw.
            mirror.apply(data)
            mujoco.mj_forward(model, data)
        elif running:
            for _ in range(10):
                mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)

        if free_cam:
            az, el, dist = cs["az"], cs["el"], cs["dist"]
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            cam.lookat[:] = target
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            # replicate the free camera pose for gsplat
            ar, er = np.deg2rad(az), np.deg2rad(el)
            fwd = np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])
            eye = target - dist * fwd
            zc = fwd / np.linalg.norm(fwd)
            xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = xc, yc, zc
            c2w[:3, 3] = eye
        else:
            c2w = cap_T.copy()          # OpenCV c2w straight from gravity_alignment
            renderer.update_scene(data, camera="capture", scene_option=vopt)

        fg = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam if free_cam else "capture", scene_option=vopt)
        seg = renderer.render()[:, :, 0].astype(np.int32)
        renderer.disable_segmentation_rendering()
        seg_buf = seg
        ui["cam_R"] = c2w[:3, :3]
        dist_now[0] = dist
        mask = np.isin(seg, keep).astype(np.float32)[..., None]

        with torch.no_grad():
            colors, _, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=g["sh"], sh_degree=sh_deg,
                viewmats=torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev)[None],
                Ks=K[None], width=a.width, height=a.height, packed=False)
        bg = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        img = {0: bg * (1 - mask) + fg * mask, 1: bg, 2: fg}[mode].astype(np.uint8)
        status = (f"MIRROR {mirror.got} frames" if mirror is not None
                  else ('RUN' if running else 'PAUSE'))
        cv2.putText(img, f"{status}  mode={['composite','bg','fg'][mode]}"
                     f"  |  drag objects with the left mouse button",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("MuJoCo + 3DGS", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        elif k == 32: running = not running
        elif k in (ord('r'), ord('R')): mujoco.mj_resetData(model, data); running = False
        elif k in (ord('b'), ord('B')): mode = (mode + 1) % 3
        elif k in (ord('c'), ord('C')): free_cam = (not free_cam) if cap_T is not None else True
        elif k == 81: cs["az"] -= 3
        elif k == 83: cs["az"] += 3
        elif k == 82: cs["el"] = min(cs["el"] + 3, 89)
        elif k == 84: cs["el"] = max(cs["el"] - 3, -89)
        elif k in (ord('='), ord('+')): cs["dist"] = max(0.15, cs["dist"] / 1.12)
        elif k in (ord('-'), ord('_')): cs["dist"] = min(60.0, cs["dist"] * 1.12)
        elif k in (ord('w'), ord('W')): target[2] += 0.05
        elif k in (ord('s'), ord('S')): target[2] -= 0.05
        elif k in (ord('x'), ord('X')):
            # swing 180 deg round the target - the fastest way to get from
            # behind the robot to facing it across the table
            cs["az"] += 180.0
            print(f"[viewer] azimuth flipped to {cs['az']:+.1f} deg")
        elif k in (ord('f'), ord('F')):
            c, d = frame_scene(); target[:] = c
            cs["dist"] = d
            print(f"[viewer] re-framed on ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
                  f"at {d:.2f} m")

    renderer.close(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
