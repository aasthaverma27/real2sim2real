#!/usr/bin/env python3
"""Serve the robot's VLA cameras from the RoboSnap scene, photoreal, over ZMQ.

This is the one piece the reference data-collection stack has no substitute for.
`gr00t_wbc`'s exporter and inference client both read images through
`VLAImageClient`, which subscribes to a camera server by *topic name*:

    stereo/right -> ego_view    (the main ego view)
    hand_left    -> ego_left    (left wrist)
    hand_right   -> ego_right   (right wrist)

On the real robot those come from the camera server on the robot; in the
reference sim they come from gear_sonic's own publisher, whose scene is the
gear lab and not a reconstruction. `run_mujoco_wbc_sim.py` publishes no images
at all, so the exporter records `image False` forever against our sim. This
fills that hole and renders those three cameras through MuGS instead, so what
lands in the dataset is the reconstructed room rather than grey MuJoCo - which
is the entire point of recording in a RoboSnap scene.

    PUB tcp://*:5555
      topic "stereo/right"  header JSON + JPEG   <- ego camera
      topic "hand_left"     header JSON + JPEG   <- left wrist camera
      topic "hand_right"    header JSON + JPEG   <- right wrist camera

Wire format is `[topic, header json, jpeg]`, three parts, exactly what
`VLAImageClient._loop` expects; a single-part msgpack server (gear_sonic's
`sensor_server.py`) is what it reports as "<1-part message>". Frames are
encoded BGR because the client decodes with `cv2.imdecode` and converts to RGB
itself. All three cameras are rendered from ONE simulator state and share a
frame id and timestamp, so a recorded step is internally consistent.

The state comes from the DDS sim through `wbc_state_relay.py`, the same way the
hybrid viewer and the ELP pair get it, so the robot and the objects here are
the ones the WBC is actually driving. Give it its own relay port: two sockets
cannot bind the same one.

    scripts/realsim/vla_camera_server.py --scene-dir outputs/myroom \
        --mjcf outputs/myroom/mujoco/scene_g1.xml \
        --ply outputs/myroom/gs_bg_notripod.ply --mirror 5603

Check it with the reference stack's own probe, from inside the container:

    python zmq_probe.py --connect tcp://localhost:5555 --seconds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Source topic -> which scene camera renders it. The keys are fixed by
# gr00t_wbc.control.sensor.vla_image_client.VLA_DATA_TOPIC_MAP; renaming one
# here means the exporter silently never completes a frame.
TOPICS = ("stereo/right", "hand_left", "hand_right")

# The other stack names the same three cameras differently: gear_sonic's sim
# publisher keys them by dataset feature name, and its client looks those up
# verbatim (base_sim.py's camera_configs). Same images, same order.
MSGPACK_KEYS = {"stereo/right": "ego_view",
                "hand_left": "ego_left",
                "hand_right": "ego_right"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--prefix", default="g1_", help="robot namespace in --mjcf")
    p.add_argument("--ego-camera", default="head_pinhole",
                   help="camera for stereo/right, without the prefix")
    p.add_argument("--left-camera", default="ego_view_left_mono")
    p.add_argument("--right-camera", default="ego_view_right_mono")
    p.add_argument("--port", type=int, default=5555,
                   help="the port the exporter's --camera_port must match")
    p.add_argument("--bind", default="tcp://*")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    # The real head stream is published 768x432 and only then resized to
    # 640x480 by the VLA client, which squashes 16:9 into 4:3 anisotropically:
    # after that squash the horizontal and vertical focal lengths differ, so a
    # single 4:3 pinhole cannot reproduce it. Render native, resize after -
    # exactly the real chain. Matches head_pinhole's fovy/resolution in the MJCF.
    p.add_argument("--ego-render-width", type=int, default=768)
    p.add_argument("--ego-render-height", type=int, default=432)
    # The wrist lenses are raw fisheye on the robot (robot-camera-server's
    # config.yaml: hand_left/hand_right 1280x720, crop 1.0, no undistort), and a
    # MuJoCo pinhole cannot imitate one - see scripts/realsim/wrist_fisheye.py.
    p.add_argument("--wrist-projection", choices=["pinhole", "fisheye"],
                   default="pinhole",
                   help="'fisheye' reproduces the real lens (equidistant, "
                        "r=f*theta) by cube-rendering each wrist and warping; "
                        "'pinhole' is the plain MuJoCo camera. Default pinhole "
                        "ONLY because fisheye costs 3 renders per wrist and the "
                        "server already runs at ~18 fps against a 20 fps "
                        "budget - turn it on when the frame budget allows, it "
                        "is the physically correct one.")
    p.add_argument("--wrist-fov-h", type=float, default=121.28,
                   help="horizontal coverage of the real wrist lens in degrees; "
                        "121.28 is what the MJCF's fovy=90 at 1280x720 covers, "
                        "which is how those cameras were aimed")
    p.add_argument("--wrist-render-width", type=int, default=1280)
    p.add_argument("--wrist-render-height", type=int, default=720,
                   help="native wrist stream size; the frame is rendered here "
                        "and resized to --width/--height afterwards, the same "
                        "chain the real camera goes through")
    p.add_argument("--wrist-face-size", type=int, default=0,
                   help="cube-face render size, 0 = 2*f (no resampling loss). "
                        "Lower it to trade wrist sharpness for frame rate.")
    p.add_argument("--probe-camera", default="scene_view",
                   help="a worldbody camera the fisheye path borrows to point "
                        "at each cube face")
    p.add_argument("--fps", type=float, default=20.0,
                   help="match the exporter's --data_collection_frequency")
    p.add_argument("--jpeg-quality", type=int, default=80,
                   help="80 is what gear_sonic's server uses")
    p.add_argument("--mode", choices=["hybrid", "mujoco_only", "3dgs_only"],
                   default="hybrid")
    p.add_argument("--protocol", choices=["both", "multipart", "msgpack"],
                   default="msgpack",
                   help="'multipart' is [topic, meta, jpeg], read by topic-aware "
                        "consumers such as vla_camera_viewer. 'msgpack' is the "
                        "single-part {timestamps, images} with base64 JPEGs "
                        "that ComposedCameraClientSensor wants -- both "
                        "gear_sonic's and gr00t_wbc's. NOTE: 'both' does NOT "
                        "mean either side works. gr00t_wbc's SensorClient does "
                        "a single-frame recv() and cannot skip the multipart "
                        "frames, so 'both' makes it die with msgpack "
                        "ExtraData. Default is 'msgpack' because that is what "
                        "the inference clients need; pass --protocol "
                        "multipart (or both) only for topic-aware consumers.")
    p.add_argument("--exclude-geoms", default="floor")
    p.add_argument("--frames", type=int, default=0, help="0 = until killed")
    p.add_argument("--mirror-frame", choices=["auto", "identity"], default="auto")
    p.add_argument("--mirror", type=int, nargs="?", const=5603, metavar="PORT",
                   help="follow the DDS sim over UDP instead of stepping "
                        "physics here. Use a port no other consumer binds.")
    p.add_argument("--no-mirror-objects", dest="mirror_objects",
                   action="store_false")
    p.add_argument("--visual-dr", nargs="?", const="", default=None,
                   metavar="CONFIG",
                   help="apply the sim's visual-randomization seed (needs the "
                        "sim started with --visual-dr and --mirror here). "
                        "CONFIG defaults to configs/visual_randomization.yaml")
    p.add_argument("--visual-report", default="",
                   help="comma-separated UDP ports to send each look's summary "
                        "to - collection.py's object port, so it can print the "
                        "lighting next to the save/discard prompt")
    p.add_argument("--visual-log", type=Path,
                   default=ROOT / "outputs" / "visual_dr" / "episodes.jsonl",
                   help="one JSON line per applied look, with its seed")
    a = p.parse_args()

    import base64

    import cv2
    import mujoco
    import zmq
    from mugs.sensors import GaussianSensor, GaussianSensorConfig

    msgpack = None
    if a.protocol in ("both", "msgpack"):
        import msgpack

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cams = {"stereo/right": a.ego_camera,
            "hand_left": a.left_camera,
            "hand_right": a.right_camera}
    names, ids = {}, {}
    for topic, base in cams.items():
        # The render scene namespaces the robot (g1_ego_view_left_mono); the
        # WBC scene does not. Accept either so one flag set works on both.
        for cand in (f"{a.prefix}{base}", base):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cand)
            if cid >= 0:
                names[topic], ids[topic] = cand, cid
                break
        else:
            have = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
                    for c in range(model.ncam)]
            raise SystemExit(f"no camera {base!r} (or {a.prefix}{base!r}) in "
                             f"{a.mjcf}\nthis scene has: {', '.join(have)}")

    exclude = {x.strip() for x in a.exclude_geoms.split(",") if x.strip()}
    fg = [g for g in range(model.ngeom)
          if int(model.geom_bodyid[g]) != 0 and int(model.geom_group[g]) != 3
          and not any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                      for x in exclude)]

    def make_sensor(w, h):
        s = GaussianSensor(GaussianSensorConfig(
            width=w, height=h, background_ply_path=a.ply,
            render_mode=a.mode, robot_geom_ids=fg, use_sh=True,
            depth_composite=True, camera_convention="mujoco"))
        s._get_renderer(model)
        s._scene_option.sitegroup[:] = 0      # camera markers are sites, not geoms
        return s

    ego_native = (a.ego_render_width, a.ego_render_height)
    # Each sensor holds its own copy of the splat, so only build the sizes that
    # are actually rendered: the fisheye path never renders a wrist at --width.
    sensor = (None if a.wrist_projection == "fisheye"
              and ego_native != (a.width, a.height) else
              make_sensor(a.width, a.height))
    ego_sensor = (sensor if ego_native == (a.width, a.height)
                  else make_sensor(*ego_native))

    WRIST_TOPICS = ("hand_left", "hand_right")
    lens = face_sensor = None
    probe_name = probe_id = None
    if a.wrist_projection == "fisheye":
        from wrist_fisheye import WristFisheye
        probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                     a.probe_camera)
        if probe_id < 0 or int(model.cam_bodyid[probe_id]) != 0:
            raise SystemExit(f"--probe-camera {a.probe_camera!r} must exist and "
                             f"be on the worldbody; the fisheye path repoints it")
        probe_name = a.probe_camera
        lens = WristFisheye(a.wrist_render_width, a.wrist_render_height,
                            a.wrist_fov_h, a.wrist_face_size)
        face_sensor = make_sensor(lens.face_size, lens.face_size)

    mirror = None
    if a.mirror:
        from wbc_state_relay import MirrorReceiver
        mirror = MirrorReceiver(model, a.scene_dir, a.prefix, a.mirror,
                                a.mirror_frame, objects=a.mirror_objects)
        print(f"[vla-cam] mirroring the DDS sim on udp {a.mirror}", flush=True)

    visual = None
    if a.visual_dr is not None:
        from visual_randomizer import VisualDRFollower
        # This server is what the exporter records, so it is the one that logs.
        visual = VisualDRFollower(model, a.visual_dr or None,
                                  [sensor, ego_sensor, face_sensor],
                                  log=a.visual_log, tag="vla_cam",
                                  report_ports=[int(x) for x in
                                                a.visual_report.split(",") if x])
        print(f"[vla-cam] visual randomization on, log {a.visual_log}", flush=True)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)             # newest frames win, never block
    endpoint = f"{a.bind}:{a.port}"
    pub.bind(endpoint)

    print(f"[vla-cam] PUB {endpoint}")
    for t in TOPICS:
        print(f"[vla-cam]   topic {t:<13} <- camera {names[t]}")
    if a.protocol in ("both", "msgpack"):
        print(f"[vla-cam]   msgpack keys "
              f"{', '.join(MSGPACK_KEYS[t] for t in TOPICS)} (gear_sonic clients)")
    print(f"[vla-cam]   head rendered {ego_native[0]}x{ego_native[1]} native "
          f"-> resized to {a.width}x{a.height} (real chain's squash)")
    if lens is not None:
        print(f"[vla-cam]   wrists FISHEYE {lens.describe()} -> resized to "
              f"{a.width}x{a.height}")
    else:
        print(f"[vla-cam]   wrists pinhole {a.width}x{a.height} (the real "
              f"lenses are fisheye; --wrist-projection fisheye matches them)")
    print(f"[vla-cam] {a.width}x{a.height} jpeg q{a.jpeg_quality} at {a.fps:g} fps, "
          f"mode {a.mode}, protocol {a.protocol}")
    print(f"[vla-cam] point the exporter at --camera_host localhost "
          f"--camera_port {a.port}", flush=True)

    period = 1.0 / a.fps if a.fps > 0 else 0.0
    frame, sent = 0, 0
    t_report = time.perf_counter()
    try:
        while a.frames == 0 or frame < a.frames:
            t0 = time.perf_counter()
            if mirror is not None:
                mirror.apply(data)            # the DDS sim owns the physics
                if visual is not None:
                    visual.update(mirror)     # new look on each sim reset
            mujoco.mj_forward(model, data)

            # One state, three views: a recorded step must not mix timesteps.
            stamp = time.time_ns()
            if lens is None:
                imgs = {t: sensor.render(model, data, names[t])
                        for t in WRIST_TOPICS}
            else:
                # Cube-render each wrist at the lens's native size and warp,
                # then resize once - the real camera's own chain.
                imgs = {}
                for t in WRIST_TOPICS:
                    w = lens.render(face_sensor, model, data, ids[t],
                                    probe_name, probe_id, cv2)
                    imgs[t] = cv2.resize(w, (a.width, a.height),
                                         interpolation=cv2.INTER_AREA)
            # Render the head at its native published size, then resize down -
            # the anisotropic squash the real VLA client applies.
            ego = ego_sensor.render(model, data, names["stereo/right"])
            if ego.shape[1::-1] != (a.width, a.height):
                ego = cv2.resize(ego, (a.width, a.height),
                                 interpolation=cv2.INTER_AREA)
            imgs["stereo/right"] = ego

            jpegs = {}
            for topic, img in imgs.items():
                # BGR on the wire: both clients hand the bytes to cv2.imdecode,
                # which returns BGR, and convert to RGB themselves.
                ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                                       [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
                if not ok:
                    continue
                jpegs[topic] = buf.tobytes()

            if a.protocol in ("both", "multipart"):
                for topic, payload in jpegs.items():
                    header = {"topic": topic, "camera": names[topic],
                              "frame_id": frame, "stamp_ns": stamp,
                              "width": a.width, "height": a.height,
                              "encoding": "jpeg", "nbytes": len(payload)}
                    pub.send_multipart([topic.encode(),
                                        json.dumps(header).encode(), payload],
                                       flags=zmq.NOBLOCK if frame else 0)
                    sent += 1

            if a.protocol in ("both", "msgpack") and jpegs:
                # ImageMessageSchema.serialize's shape: base64 JPEG strings
                # under "images", wall-clock seconds under "timestamps". The
                # sim publisher also repeats each image at the top level, so
                # do the same - a consumer reading either place works.
                t_s = stamp / 1e9
                b64 = {MSGPACK_KEYS[t]: base64.b64encode(p).decode("utf-8")
                       for t, p in jpegs.items()}
                msg = {"timestamps": {k: t_s for k in b64}, "images": dict(b64)}
                msg.update(b64)
                try:
                    pub.send(msgpack.packb(msg, use_bin_type=True),
                             flags=zmq.NOBLOCK if frame else 0)
                    sent += 1
                except zmq.Again:
                    pass
            frame += 1

            now = time.perf_counter()
            if now - t_report > 5.0:
                got = f", {mirror.got} state / {mirror.got_obj} object pkts" \
                      if mirror is not None else ""
                print(f"[vla-cam] frame {frame}, {sent} msgs, "
                      f"{1 / max(now - t0, 1e-9):.1f} fps instantaneous{got}",
                      flush=True)
                t_report = now
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(0)
        ctx.term()
        for s in {id(x): x for x in (sensor, ego_sensor, face_sensor)
                  if x is not None}.values():
            s.close()
        print(f"[vla-cam] stopped after {frame} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
