#!/usr/bin/env python3
"""Render the G1's ELP fisheye eyes - MuJoCo has no fisheye projection.

Every MuJoCo camera is a rectilinear pinhole. The ELP-USB3DGS1200P01-L180 is a
~170 deg horizontal fisheye, and a 170 deg rectilinear render is not a wide
version of the same picture - it is a different projection, stretched without
bound towards the edges. Setting fovy=140 and calling it done would produce an
image the real camera can never take, which is exactly the kind of gap that
breaks sim2real.

So this renders the scene as a cube map from each eye pose - five 90 deg pinhole
faces, which is all that is needed for rays within 90 deg of forward - and
remaps them into an equidistant fisheye (r = f*theta), the standard model for
this class of lens. Each face goes through MuGS, so the Gaussian background is
in the result just as it is in the pinhole views.

    scripts/fisheye_render.py --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
        --ply outputs/g1_ok/bg_lm.ply --eye both -o /tmp/elp.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))

# Columns are the face's (right, up, backward) axes written in the eye's own
# camera frame, which MuJoCo orients as x right, y up, -z forward.
def _face(x, y, z):
    return np.column_stack([np.array(x, float), np.array(y, float), np.array(z, float)])


FACES = {
    "front": _face((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "right": _face((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "left":  _face((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "up":    _face((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "down":  _face((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}
for _k, _M in FACES.items():                       # right-handed sanity check
    assert np.allclose(np.cross(_M[:, 0], _M[:, 1]), _M[:, 2], atol=1e-9), _k


def fisheye_rays(w: int, h: int, fov_h_deg: float):
    """Equidistant fisheye ray directions, in the eye's camera frame.

    f comes from the horizontal FOV because that is the number lens makers
    quote; the vertical coverage that falls out is reported, not assumed.
    """
    theta_max = np.deg2rad(fov_h_deg) / 2
    f = (w / 2) / theta_max
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dx = u - w / 2
    dy = h / 2 - v                                  # image y is down, camera y is up
    r = np.hypot(dx, dy)
    theta = r / f
    valid = theta < np.deg2rad(89.0)                # a cube map cannot go past 90
    with np.errstate(invalid="ignore", divide="ignore"):
        cx, sy = dx / np.maximum(r, 1e-9), dy / np.maximum(r, 1e-9)
    st = np.sin(theta)
    d = np.stack([st * cx, st * sy, -np.cos(theta)], -1)
    d[r < 1e-9] = [0.0, 0.0, -1.0]
    vert = 2 * np.rad2deg((h / 2) / f)
    return d, valid, f, vert


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--eye", default="both",
                   choices=["elp_left", "elp_right", "both"])
    p.add_argument("--prefix", default="g1_", help="prefix add_robot.py applied")
    p.add_argument("--probe-camera", default="scene_view",
                   help="a worldbody camera to borrow for the cube faces")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--fov-h", type=float, default=170.0)
    p.add_argument("--face-size", type=int, default=768)
    p.add_argument("--exclude-geoms", default="floor")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--live", action="store_true",
                   help="open a window and keep re-rendering; SPACE runs "
                        "physics, ESC quits. Ten MuGS renders per frame (five "
                        "cube faces per eye), so expect a couple of FPS.")
    p.add_argument("--pos", type=int, nargs=2, default=None)
    p.add_argument("--topmost", action="store_true",
                   help="keep the window above other windows")
    a = p.parse_args()

    import cv2
    import mujoco
    from mugs.sensors import GaussianSensor, GaussianSensorConfig

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.probe_camera)
    if probe < 0 or int(model.cam_bodyid[probe]) != 0:
        raise SystemExit(f"--probe-camera {a.probe_camera!r} must be on the worldbody")

    exclude = {x.strip() for x in a.exclude_geoms.split(",") if x.strip()}
    fg = [g for g in range(model.ngeom)
          if int(model.geom_bodyid[g]) != 0 and int(model.geom_group[g]) != 3
          and not any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                      for x in exclude)]

    S = a.face_size
    sensor = GaussianSensor(GaussianSensorConfig(
        width=S, height=S, background_ply_path=a.ply, render_mode="hybrid",
        robot_geom_ids=fg, use_sh=True, depth_composite=True,
        camera_convention="mujoco"))
    f_face = S / 2.0                                # 90 deg square face

    # Sites are debug decoration, not scene content. The head camera sits 3 cm
    # from its own mounting site, so an undisabled site sphere fills a chunk of
    # a 170 deg frame.
    sensor._get_renderer(model)
    sensor._scene_option.sitegroup[:] = 0

    eyes = ([a.prefix + "elp_left", a.prefix + "elp_right"]
            if a.eye == "both" else [a.prefix + a.eye])
    dirs, valid, f_px, vert = fisheye_rays(a.width, a.height, a.fov_h)
    print(f"[fisheye] equidistant f={f_px:.1f} px/rad from {a.fov_h} deg horizontal"
          f" -> {vert:.1f} deg vertical over {a.height} px")

    def render_eyes():
        outs = []
        for eye in eyes:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, eye)
            if cid < 0:
                raise SystemExit(f"no camera {eye!r}; is the ELP module in this MJCF?")
            eye_pos = data.cam_xpos[cid].copy()
            eye_R = data.cam_xmat[cid].reshape(3, 3).copy()

            img = np.zeros((a.height, a.width, 3), np.uint8)
            filled = np.zeros((a.height, a.width), bool)
            for name, M in FACES.items():
                model.cam_pos[probe] = eye_pos
                R = eye_R @ M
                q = np.empty(4); mujoco.mju_mat2Quat(q, R.reshape(9))
                model.cam_quat[probe] = q
                model.cam_fovy[probe] = 90.0
                mujoco.mj_forward(model, data)
                face = sensor.render(model, data, a.probe_camera,
                                     camera_params={"position": data.cam_xpos[probe].copy(),
                                                    "rotation": data.cam_xmat[probe].reshape(3, 3).copy(),
                                                    "fx": f_face, "fy": f_face,
                                                    "width": S, "height": S})

                df = dirs @ M                            # eye frame -> face frame
                fwd = -df[:, :, 2]
                with np.errstate(invalid="ignore", divide="ignore"):
                    xs = df[:, :, 0] / fwd
                    ys = df[:, :, 1] / fwd
                take = valid & ~filled & (fwd > 1e-6) & (np.abs(xs) <= 1) & (np.abs(ys) <= 1)
                if not take.any():
                    continue
                mx = (f_face * xs + S / 2).astype(np.float32)
                my = (S / 2 - f_face * ys).astype(np.float32)
                warped = cv2.remap(face, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                img[take] = warped[take]
                filled |= take
                if not a.live:
                    print(f"[fisheye] {eye} {name:5s} face -> "
                          f"{100 * take.mean():5.1f}% of the frame")

            if not a.live:
                print(f"[fisheye] {eye}: {100 * filled.mean():.1f}% covered "
                      f"({100 * (valid & ~filled).mean():.1f}% unfilled inside "
                      f"the lens circle)")
            outs.append(img)

        return np.concatenate(outs, 1) if len(outs) > 1 else outs[0]

    if a.live:
        win = "ELP-USB3DGS1200P01-L180  fisheye stereo"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1600, a.width * len(eyes)), a.height)
        if a.pos:
            cv2.moveWindow(win, a.pos[0], a.pos[1])
        if a.topmost:
            # there is no window manager tool on the headless display, so the
            # window has to raise itself or it opens behind the editor
            try:
                cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1.0)
            except Exception as exc:
                print(f"[fisheye] could not set topmost: {exc}")
        running = False
        while True:
            if running:
                for _ in range(10):
                    mujoco.mj_step(model, data)
            img = render_eyes()
            disp = np.ascontiguousarray(img[:, :, ::-1])
            cv2.putText(disp, f"ELP fisheye {a.fov_h:.0f} deg h  "
                              f"{'RUN' if running else 'PAUSE'}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == 32:
                running = not running
            if k in (ord('r'), ord('R')):
                mujoco.mj_resetData(model, data); running = False
        cv2.destroyAllWindows()
        sensor.close()
        return 0

    out = render_eyes()
    cv2.imwrite(str(a.output), out[:, :, ::-1])
    print(f"[fisheye] wrote {a.output}  {out.shape[1]}x{out.shape[0]}")
    sensor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
