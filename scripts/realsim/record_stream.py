#!/usr/bin/env python3
"""Record a published camera topic to a video file. Read-only insurance.

The interactive calibration tool keeps every captured frame in memory and
writes nothing until it is told to save - so a missed keypress, a crash, or a
Ctrl-C loses the whole session. This is a second subscriber that quietly
writes the stream to disk while you work. If the save is lost, the
calibration can still be recovered:

    opencv_interactive-calibration -v=<file> -t=chessboard -w=8 -h=6 --sz=27.94
    opencv_tutorial_calibration.py  (after extracting frames)

It does not touch the bridge, the tool, or the camera.

    record_stream.py --topic stereo/right -o calib/rec_stereo_right.avi
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", default=__import__("os").environ.get("RS_ROBOT_CAMERA", "tcp://192.168.123.164:5555"))
    p.add_argument("--topic", default="stereo/right")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=0.0, help="0 = measure it")
    a = p.parse_args()

    import cv2
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 8)
    sock.connect(a.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, a.topic)
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    stamps, first = [], None
    while len(stamps) < 20:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            print(f"[rec] no frames on {a.topic}", file=sys.stderr)
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
    fps = a.fps if a.fps > 0 else float(1.0 / np.median(dt)) if dt.size else 15.0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    # MJPG in AVI: every frame is a keyframe, so a file cut short by a Ctrl-C
    # is still readable up to the last frame written. That is the whole point.
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"MJPG"),
                         fps, (w, h))
    if not vw.isOpened():
        print(f"[rec] cannot open {a.out} for writing", file=sys.stderr)
        return 1
    print(f"[rec] {a.topic} {w}x{h} @ {fps:.1f} fps -> {a.out}", flush=True)
    print("[rec] recording. Ctrl-C when the calibration is done.", flush=True)

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
            vw.write(img)
            n += 1
            if n % 100 == 0:
                mb = a.out.stat().st_size / 1e6 if a.out.exists() else 0
                print(f"\r[rec] {n} frames, {mb:.0f} MB", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        vw.release()
        mb = a.out.stat().st_size / 1e6 if a.out.exists() else 0
        print(f"\n[rec] saved {n} frames, {mb:.0f} MB -> {a.out}")
    return 0


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
    me = Path(sys.executable).resolve()
    for cand in _FALLBACK_PYTHONS:
        if not Path(cand).exists() or Path(cand).resolve() == me:
            continue
        if subprocess.run([cand, "-c", "import cv2, zmq"],
                          capture_output=True).returncode == 0:
            os.execv(cand, [cand, str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit("need opencv-python and pyzmq")


if __name__ == "__main__":
    ensure_deps()
    sys.exit(main())
