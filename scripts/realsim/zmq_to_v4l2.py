#!/usr/bin/env python3
"""Bridge one published camera topic into a V4L2 device.

opencv_interactive-calibration - the tool that shows you where to move the
board and tells you when it has enough - reads a camera by index or a video
file. It cannot subscribe to ZMQ. But the frame that has to be calibrated is
the PUBLISHED one: the robot's server has already cropped and downscaled it,
and a model fitted to anything else does not describe what the VLA sees.

So this puts the published stream behind a /dev/video node, and the OpenCV
tool sees an ordinary webcam. Frames go out at whatever rate they arrive, so
the frame rate the tool displays is the stream's real one.

    # once, needs root:
    sudo modprobe v4l2loopback video_nr=9 card_label=robot_cam exclusive_caps=1

    # then:
    zmq_to_v4l2.py --topic stereo/right --device /dev/video9
    opencv_interactive-calibration --ci=9 -t chessboard -w 8 -h 6 --sz 27.94
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", default=__import__("os").environ.get("RS_ROBOT_CAMERA", "tcp://192.168.123.164:5555"))
    p.add_argument("--topic", default="stereo/right",
                   help="stereo/right is the VLA's ego_view")
    p.add_argument("--device", default="/dev/video9")
    p.add_argument("--fps", type=float, default=0.0,
                   help="declared output rate; 0 measures it from the stream")
    a = p.parse_args()

    import cv2
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.connect(a.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, a.topic)
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    print(f"[bridge] waiting for {a.topic} on {a.connect}", flush=True)

    # Measure the stream before declaring a format: ffmpeg needs the size and
    # rate up front, and guessing either makes the tool's own fps read wrong.
    stamps, first = [], None
    while len(stamps) < 20:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            print(f"[bridge] no frames on {a.topic} - is the server up?",
                  file=sys.stderr)
            return 1
        if len(parts) < 2:
            continue
        img = cv2.imdecode(np.frombuffer(parts[-1], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        first = img
        stamps.append(time.time())

    h, w = first.shape[:2]
    dt = np.diff(np.asarray(stamps))
    dt = dt[dt > 0]
    measured = float(1.0 / np.median(dt)) if dt.size else 15.0
    fps = a.fps if a.fps > 0 else round(measured, 3)
    print(f"[bridge] {a.topic}: {w}x{h} at {measured:.1f} fps -> {a.device}",
          flush=True)

    holder = subprocess.run(["lsof", "-t", a.device], capture_output=True,
                            text=True).stdout.split()
    if holder:
        who = subprocess.run(["ps", "-o", "pid=,args=", *holder],
                             capture_output=True, text=True).stdout.strip()
        print(f"[bridge] {a.device} is already held by another writer:\n"
              f"{who}\n"
              f"[bridge] stop that one first - v4l2loopback cannot change frame "
              f"size while it is open.\n"
              f"[bridge]   kill {' '.join(holder)}", file=sys.stderr)
        return 1

    cmd = ["ffmpeg", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
           "-f", "v4l2", "-pix_fmt", "yuv420p", a.device]
    try:
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        print("[bridge] ffmpeg not found", file=sys.stderr)
        return 1

    idx = a.device.replace("/dev/video", "")
    out = f"calib/{a.topic.replace('/', '_')}_opencv.xml"
    print(f"[bridge] running. Now start:\n"
          f"    opencv_interactive-calibration --ci={idx} -t=chessboard "
          f"-w=8 -h=6 --sz=27.94 --of={out}\n"
          f"  (every flag needs '=', OpenCV's parser ignores space-separated "
          f"values)\n"
          f"  Ctrl-C here when the calibration is saved.", flush=True)

    n = 0
    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) < 2:
                continue
            img = cv2.imdecode(np.frombuffer(parts[-1], np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            ff.stdin.write(img.tobytes())
            n += 1
            if n % 150 == 0:
                print(f"\r[bridge] {n} frames", end="", flush=True)
    except (KeyboardInterrupt, BrokenPipeError):
        print("\n[bridge] stopping")
    finally:
        try:
            ff.stdin.close()
            ff.wait(timeout=3)
        except Exception:
            ff.kill()
    return 0


# The shell this is launched from is usually conda `base`, which has no cv2.
# Hand over to an interpreter that has one rather than fail on the import.
# $RS_PY_SIM (configs/paths.env) has cv2, yaml and zmq.
_FALLBACK_PYTHONS = tuple(p for p in (__import__("os").environ.get("RS_PY_SIM"),
                                      "/usr/bin/python3") if p)


def ensure_deps() -> None:
    try:
        import cv2, zmq  # noqa: F401
        return
    except ImportError:
        pass
    import os
    me = __import__("pathlib").Path(sys.executable).resolve()
    for cand in _FALLBACK_PYTHONS:
        c = __import__("pathlib").Path(cand)
        if not c.exists() or c.resolve() == me:
            continue
        if subprocess.run([cand, "-c", "import cv2, zmq"],
                          capture_output=True).returncode == 0:
            print(f"[bridge] {sys.executable} has no cv2/pyzmq - re-running "
                  f"under {cand}", file=sys.stderr, flush=True)
            os.execv(cand, [cand, str(__import__("pathlib").Path(__file__).resolve()),
                            *sys.argv[1:]])
    raise SystemExit("need opencv-python and pyzmq")


if __name__ == "__main__":
    ensure_deps()
    sys.exit(main())
