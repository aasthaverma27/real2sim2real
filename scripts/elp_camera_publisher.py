#!/usr/bin/env python3
"""Publish the G1's ELP fisheye eyes over ZMQ, one stream per lens.

The point is that nothing downstream should be able to tell this apart from the
real ELP-USB3DGS1200P01-L180. That module presents two synchronous global-shutter
streams, so this publishes two topics that share a frame index and a capture
timestamp - not one side-by-side image that a consumer has to slice, and not a
GUI window.

    PUB tcp://*:5560
      topic "elp/left"   header JSON + JPEG (or raw BGR)
      topic "elp/right"  same frame_id, same stamp

Each message is multipart: [topic, header json, payload]. The header carries
everything a stereo consumer needs - frame_id, stamp_ns, width, height,
encoding, the fisheye model and its focal length in px/rad, and the baseline -
so a subscriber never has to hardcode the intrinsics.

    scripts/elp_camera_publisher.py --mjcf outputs/g1_ok/mujoco/scene_g1.xml \
        --ply outputs/g1_ok/bg_lm.ply --port 5560 --fps 10
    scripts/elp_camera_subscriber.py --port 5560          # to check it

Projection: MuJoCo has no fisheye, so each eye is rendered as a five-face cube
map and remapped to an equidistant fisheye (r = f*theta). See
scripts/fisheye_render.py, which this imports rather than duplicating.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--prefix", default="g1_")
    p.add_argument("--probe-camera", default="scene_view")
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--bind", default="tcp://*")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fov-h", type=float, default=170.0)
    p.add_argument("--face-size", type=int, default=640)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--encoding", choices=["jpeg", "bgr8"], default="jpeg")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--physics", action="store_true",
                   help="step the simulation between frames")
    p.add_argument("--frames", type=int, default=0, help="0 = run until killed")
    p.add_argument("--exclude-geoms", default="floor")
    p.add_argument("--scene-dir", type=Path, default=None)
    p.add_argument("--mirror-frame", choices=["auto", "identity"], default="auto",
                   help="'auto' undoes build_wbc_scene's re-origin when "
                        "scene_wbc.frame.json exists; 'identity' when the "
                        "publisher renders the very same scene file.")
    p.add_argument("--mirror", type=int, nargs="?", const=5601, metavar="PORT",
                   help="take the robot's pose from the running DDS sim instead "
                   "of stepping physics here, so the fisheye pair agrees with "
                   "the sim the WBC is actually driving. Needs "
                   "scripts/wbc_state_relay.py running in .venv_sim.")
    a = p.parse_args()

    import cv2
    import mujoco
    import zmq
    from mugs.sensors import GaussianSensor, GaussianSensorConfig
    from fisheye_render import FACES, fisheye_rays

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.probe_camera)
    if probe < 0 or int(model.cam_bodyid[probe]) != 0:
        raise SystemExit(f"--probe-camera {a.probe_camera!r} must be on the worldbody")

    eyes = {"left": a.prefix + "elp_left", "right": a.prefix + "elp_right"}
    ids = {}
    for side, name in eyes.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise SystemExit(f"no camera {name!r} in {a.mjcf}")
        ids[side] = cid
    baseline = float(np.linalg.norm(data.cam_xpos[ids["left"]]
                                    - data.cam_xpos[ids["right"]]))

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
    sensor._get_renderer(model)
    sensor._scene_option.sitegroup[:] = 0        # no debug markers in a camera feed
    f_face = S / 2.0

    dirs, valid, f_px, vert = fisheye_rays(a.width, a.height, a.fov_h)
    # The remap tables depend only on the lens, so build them once per face
    # instead of once per frame - they are the expensive part of the warp.
    tables = []
    for name, M in FACES.items():
        df = dirs @ M
        fwd = -df[:, :, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            xs, ys = df[:, :, 0] / fwd, df[:, :, 1] / fwd
        take = valid & (fwd > 1e-6) & (np.abs(xs) <= 1) & (np.abs(ys) <= 1)
        mx = np.nan_to_num(f_face * xs + S / 2).astype(np.float32)
        my = np.nan_to_num(S / 2 - f_face * ys).astype(np.float32)
        tables.append((name, M, take, mx, my))
    # a pixel can fall inside two faces at the seam; first face wins, as before
    claimed = np.zeros((a.height, a.width), bool)
    finals = []
    for name, M, take, mx, my in tables:
        t = take & ~claimed
        claimed |= t
        finals.append((name, M, t, mx, my))

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)                # drop old frames, never block
    endpoint = f"{a.bind}:{a.port}"
    pub.bind(endpoint)

    info = {
        "model": "ELP-USB3DGS1200P01-L180",
        "sensor": "OG02B10",
        "shutter": "global",
        "projection": "equidistant_fisheye",
        "f_px_per_rad": round(float(f_px), 3),
        "fov_h_deg": a.fov_h,
        "fov_v_deg": round(float(vert), 2),
        "baseline_m": round(baseline, 4),
        "width": a.width, "height": a.height,
        "encoding": a.encoding,
    }
    print(f"[elp] PUB {endpoint}  topics elp/left elp/right")
    print(f"[elp] {json.dumps(info)}")
    print(f"[elp] baseline {1000 * baseline:.1f} mm (measured on the real module, "
          f"lens centre to lens centre)")

    def render(side):
        cid = ids[side]
        eye_pos = data.cam_xpos[cid].copy()
        eye_R = data.cam_xmat[cid].reshape(3, 3).copy()
        img = np.zeros((a.height, a.width, 3), np.uint8)
        for name, M, take, mx, my in finals:
            if not take.any():
                continue
            model.cam_pos[probe] = eye_pos
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, (eye_R @ M).reshape(9))
            model.cam_quat[probe] = q
            model.cam_fovy[probe] = 90.0
            mujoco.mj_forward(model, data)
            face = sensor.render(model, data, a.probe_camera,
                                 camera_params={"position": data.cam_xpos[probe].copy(),
                                                "rotation": data.cam_xmat[probe].reshape(3, 3).copy(),
                                                "fx": f_face, "fy": f_face,
                                                "width": S, "height": S})
            warped = cv2.remap(face, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            img[take] = warped[take]
        return img

    mirror = None
    if a.mirror:
        from wbc_state_relay import MirrorReceiver
        mirror = MirrorReceiver(model, a.scene_dir, a.prefix, a.mirror, a.mirror_frame)
        print(f"[elp] mirroring the DDS sim on udp {a.mirror}", flush=True)

    period = 1.0 / a.fps if a.fps > 0 else 0.0
    frame = 0
    t_report = time.perf_counter()
    try:
        while a.frames == 0 or frame < a.frames:
            t0 = time.perf_counter()
            if mirror is not None:
                mirror.apply(data)              # the DDS sim owns the physics
                mujoco.mj_forward(model, data)
            elif a.physics:
                for _ in range(10):
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # both lenses are read from one simulator state, which is what
            # "synchronous dual lens" means on the real module
            stamp = time.time_ns()
            imgs = {side: render(side) for side in ("left", "right")}

            for side, img in imgs.items():
                bgr = img[:, :, ::-1]
                if a.encoding == "jpeg":
                    ok, buf = cv2.imencode(".jpg", bgr,
                                           [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
                    if not ok:
                        continue
                    payload = buf.tobytes()
                else:
                    payload = np.ascontiguousarray(bgr).tobytes()
                header = dict(info, side=side, frame_id=frame, stamp_ns=stamp,
                              nbytes=len(payload))
                pub.send_multipart([f"elp/{side}".encode(),
                                    json.dumps(header).encode(), payload],
                                   flags=zmq.NOBLOCK if frame else 0)
            frame += 1
            now = time.perf_counter()
            if now - t_report > 5.0:
                print(f"[elp] frame {frame}, {1 / max(now - t0, 1e-9):.1f} fps "
                      f"instantaneous", flush=True)
                t_report = now
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(0); ctx.term(); sensor.close()
        print(f"[elp] stopped after {frame} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
