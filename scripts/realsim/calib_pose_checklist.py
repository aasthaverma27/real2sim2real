#!/usr/bin/env python3
"""Live checklist of the board poses a calibration needs. Read-only.

opencv_interactive-calibration tells you how many frames it has, but not
WHICH views are still missing - and missing views are what makes a fit wrong
while its residual still looks healthy. This watches the same published
stream as a second subscriber and ticks off each required pose as you achieve
it, so you know what is left and when you are done.

It never touches the camera, the bridge, or the calibration tool. Run it in a
third terminal, alongside them:

    calib_pose_checklist.py --topic stereo/right

A pose counts as achieved once it has been held for a moment - long enough
that the calibration tool has had a chance to capture it too.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

import numpy as np

HOLD_S = 1.2          # hold a pose this long before it counts
# opencv_interactive-calibration rejects any view whose Euler tilt about X or Y
# exceeds 40 deg (frameProcessor.cpp, badAngleThresh). Aim well inside that:
# steep views help conditioning right up until the tool throws them away.
TILT_MIN, TILT_MAX = 20.0, 36.0
TILT_REJECT = 40.0
NEAR_FILL = 0.16      # board area fraction that counts as "close"
FAR_FILL = 0.11       # ... and as "far"
PERSP = 0.88          # an edge this much shorter than its opposite is tilted away


def edge_widths(pts):
    """Apparent length of each of the four board edges, in image space.

    Taken from the extreme rows/columns of the detected corners rather than
    from the grid's index order: an 8x6 board is symmetric under a 180 degree
    turn, so the detector's ordering can flip, and anything that trusts it
    would report 'top away' when the bottom is away. Image space cannot flip.
    """
    ys, xs = pts[:, 1], pts[:, 0]
    k = max(3, len(pts) // 8)
    top = pts[np.argsort(ys)[:k]]
    bot = pts[np.argsort(ys)[-k:]]
    left = pts[np.argsort(xs)[:k]]
    right = pts[np.argsort(xs)[-k:]]
    return (float(np.ptp(top[:, 0])), float(np.ptp(bot[:, 0])),
            float(np.ptp(left[:, 1])), float(np.ptp(right[:, 1])))


def euler_xy(cv2, corners, size, board, square):
    """|rx|, |ry| the way the calibration tool measures them.

    Its reject test is per-axis, not on total tilt, so a board slanted 30 deg
    about both axes passes while one slanted 42 deg about a single axis does
    not - even though the second looks less extreme in the frame.
    """
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    w, h = size
    K = np.array([[0.8 * w, 0, w / 2.0], [0, 0.8 * w, h / 2.0], [0, 0, 1.0]],
                 float)
    try:
        ok, r, _ = cv2.solvePnP(objp,
                                np.asarray(corners, np.float32).reshape(-1, 1, 2),
                                K, None)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(r)
    rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    ry = math.degrees(math.atan2(-R[2, 0],
                                 math.hypot(R[2, 1], R[2, 2])))
    return abs(rx), abs(ry)


def roll_deg(pts, board):
    """In-plane rotation, modulo 90 - so the 180 degree ambiguity cannot bite."""
    bw, bh = board
    try:
        grid = pts.reshape(bh, bw, 2)
    except ValueError:
        return 0.0
    v = grid[0, -1] - grid[0, 0]
    return abs(math.degrees(math.atan2(float(v[1]), float(v[0])))) % 90.0


def classify(m, tilt, pts, board):
    """Which of the required poses this frame satisfies. May be more than one."""
    cx, cy = m["cen"]
    fill = m["area"]
    tw, bw_, lh, rh = edge_widths(pts)
    roll = roll_deg(pts, board)
    centreish = 0.22 < cx < 0.78 and 0.18 < cy < 0.82
    hits = set()

    if centreish and tilt < 18 and fill >= NEAR_FILL:
        hits.add(0)
    if centreish and TILT_MIN < tilt < TILT_MAX:
        if tw < bw_ * PERSP:
            hits.add(1)
        if bw_ < tw * PERSP:
            hits.add(2)
        if lh < rh * PERSP:
            hits.add(3)
        if rh < lh * PERSP:
            hits.add(4)
    if cx < 0.36 and cy < 0.42:
        hits.add(5)
    if cx > 0.64 and cy < 0.42:
        hits.add(6)
    if cx < 0.36 and cy > 0.58:
        hits.add(7)
    if cx > 0.64 and cy > 0.58:
        hits.add(8)
    if centreish and 25 < roll < 65:
        hits.add(9)
    if fill <= FAR_FILL and tilt > 20:
        hits.add(10)
    return hits, roll, fill


POSES = [
    "centre, square to the lens, CLOSE (fills a third)",
    "centre, TOP edge tilted away ~40 deg",
    "centre, BOTTOM edge tilted away ~40 deg",
    "centre, LEFT edge tilted away ~40 deg",
    "centre, RIGHT edge tilted away ~40 deg",
    "TOP-LEFT corner of the frame, out to the edge",
    "TOP-RIGHT corner",
    "BOTTOM-LEFT corner",
    "BOTTOM-RIGHT corner",
    "centre, ROLLED ~45 deg in its own plane",
    "FAR (small in the frame), and tilted",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", default=__import__("os").environ.get("RS_ROBOT_CAMERA", "tcp://192.168.123.164:5555"))
    p.add_argument("--topic", default="stereo/right")
    p.add_argument("--board", default="8x6", help="INNER corners")
    a = p.parse_args()

    import cv2
    import zmq
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from camera_calibration import detect, frame_metrics, board_pose

    board = tuple(int(v) for v in a.board.lower().split("x"))

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.connect(a.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, a.topic)
    sock.setsockopt(zmq.RCVTIMEO, 2000)

    done = set()
    holding = {}                      # pose index -> when it started
    last_draw = 0.0
    status = "waiting for frames"
    live = {}

    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                status = "no frames - is the camera server up?"
                parts = None
            if parts and len(parts) >= 2:
                img = cv2.imdecode(np.frombuffer(parts[-1], np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    found, corners, _ = detect(cv2, gray, board)
                    if not found:
                        status = "board not visible"
                        holding.clear()
                        live = {}
                    else:
                        m = frame_metrics(cv2, gray, corners)
                        pose = board_pose(cv2, corners, gray.shape[::-1],
                                          board, 0.0279)
                        if m and pose:
                            tilt = pose[0]
                            pts = np.asarray(corners, np.float32).reshape(-1, 2)
                            hits, roll, fill = classify(m, tilt, pts, board)
                            live = dict(tilt=tilt, roll=roll, fill=fill,
                                        cen=m["cen"], hits=hits)
                            now = time.time()
                            for i in list(holding):
                                if i not in hits:
                                    holding.pop(i, None)
                            for i in hits:
                                if i in done:
                                    continue
                                holding.setdefault(i, now)
                                if now - holding[i] >= HOLD_S:
                                    done.add(i)
                                    holding.pop(i, None)
                            status = "tracking"

            if time.time() - last_draw < 0.25:
                continue
            last_draw = time.time()

            out = ["\033[H\033[J",
                   f"  BOARD POSE CHECKLIST   {a.topic}   "
                   f"{len(done)}/{len(POSES)} done"]
            if live:
                out.append(f"  tilt {live['tilt']:4.0f} deg   roll "
                           f"{live['roll']:4.0f} deg   fill "
                           f"{100*live['fill']:4.1f}%   at "
                           f"({live['cen'][0]:.2f}, {live['cen'][1]:.2f})")
            else:
                out.append(f"  {status}")
            out.append("  " + "-" * 62)
            nxt = None
            for i, name in enumerate(POSES):
                if i in done:
                    mark, note = "x", ""
                elif i in holding:
                    mark = " "
                    note = f"   HOLD {HOLD_S - (time.time()-holding[i]):.1f}s"
                else:
                    mark, note = " ", ""
                    if nxt is None:
                        nxt = i
                out.append(f"  [{mark}] {i+1:2d}  {name}{note}")
            out.append("  " + "-" * 62)
            if len(done) == len(POSES):
                out.append("  ALL POSES DONE - keep adding a few free views, "
                           "then press s in the tool")
            elif nxt is not None:
                out.append(f"  NEXT: {POSES[nxt]}")
            print("\n".join(out), flush=True)
    except KeyboardInterrupt:
        print(f"\n  stopped - {len(done)}/{len(POSES)} poses covered")
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
    from pathlib import Path
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
