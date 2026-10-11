#!/usr/bin/env python3
"""Measure the real cameras' intrinsics - properly, in ordinary room light.

This replaces the numbers in calib/head.yaml and calib/wrist_right.yaml, which
were fitted from recorded teleop frames where the board lay flat on a table.
Every view had nearly the same tilt, which makes a planar calibration
ill-conditioned: focal length and distance trade off directly, so the solver
returns a confident-looking answer that is wrong. Both files say `rms 2-4 px`
and `do not trust this fit`, and their fy/fx disagrees with the publish chain.

What is different here:

  capture      No fixed pose list. A live scoreboard over the space that
               actually sets the conditioning - 9 image zones x 3 tilt bins x
               2 distance bins - and the run ends when that space is FULL, not
               when a timer expires. You are steered into whatever cell is
               still empty. Nothing can be silently skipped.

  lighting     No Michelson-contrast gate. That threshold (0.45) is
               unreachable for a printed board under normal room light - the
               white squares clip toward 255 and the blacks lift - which is
               why the camera had to be dimmed over ssh to get views accepted.
               Detection by findChessboardCornersSB, with a CLAHE retry, IS
               the contrast test. What gets rejected instead is what genuinely
               hurts corner accuracy: specular GLARE on the paper, and MOTION.
               A lighting probe at startup reports what the room is doing
               rather than telling you to change it.

  evidence     Every detection is written to disk, accepted or not, so a bad
               fit is re-run offline with --refit instead of another twenty
               minutes holding cardboard.

  validation   rms alone provably does not work here - both bad fits reported
               healthy residuals. So: k-fold held-out reprojection, bootstrap
               confidence intervals on fx/fy/cx/cy, and a check of fy/fx
               against the ratio the publish chain forces. Each check prints
               PASS or FAIL with the number it was judged on.

Usage:

    camera_calibration.py --camera head            # both eyes at once
    camera_calibration.py --camera wrist_left
    camera_calibration.py --camera wrist_right
    camera_calibration.py --probe --camera head    # just grade the lighting
    camera_calibration.py --refit calib/capture_head_20260918_141203

The camera server stays up and nothing on the Jetson needs changing. Results
land in calib/<topic>.yaml - the name camera.py reads for `undistort: true` -
each with a `sim_camera:` block holding the MuJoCo fovy, the resolution to
render at, and the resolution to resize to.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------
# The shell this is usually launched from is conda `base`, which has no cv2.
# Rather than fail with an import error, find an interpreter that does have
# one and hand over to it - the user asked to run a script, not to go shopping
# for a python.
# $RS_PY_SIM (configs/paths.env) has cv2, yaml and zmq.
_FALLBACK_PYTHONS = tuple(p for p in (os.environ.get("RS_PY_SIM"),
                                      "/usr/bin/python3") if p)


def ensure_deps() -> None:
    try:
        import cv2, yaml, zmq  # noqa: F401
        return
    except ImportError:
        pass
    me = Path(sys.executable).resolve()
    for cand in _FALLBACK_PYTHONS:
        p = Path(cand)
        if not p.exists() or p.resolve() == me:
            continue
        probe = subprocess.run([cand, "-c", "import cv2, yaml, zmq"],
                               capture_output=True)
        if probe.returncode == 0:
            print(f"[calib] {sys.executable} has no cv2/pyzmq - re-running "
                  f"under {cand}\n", file=sys.stderr, flush=True)
            os.execv(cand, [cand, str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit("need opencv-python, pyyaml and pyzmq; none of "
                     f"{(sys.executable,) + _FALLBACK_PYTHONS} has all three")


# --------------------------------------------------------------------------
# what the robot publishes
# --------------------------------------------------------------------------
# Every stream leaves the camera server 16:9 with square pixels, and the VLA
# client then resizes it to 640x480 with a plain cv2.resize - anisotropically.
# That squash is the whole reason fy/fx is predictable, so it is written down
# once here. See configs/robots/g1_29dof_with_hand.yaml `cameras:`.
SOURCE_ASPECT = 16.0 / 9.0
VLA_OUT = (640, 480)

CAMERAS = {
    "head": dict(
        # stereo/right FIRST: it is the eye the VLA actually consumes -
        # vla_camera_server.py maps stereo/right -> ego_view - so it is the one
        # the operator must be steered by. The other eye is picked up alongside.
        topics=("stereo/right", "stereo/left"),
        model="auto",
        published=(768, 432),
        note="ELP-USB3DGS1200P01-L180 fisheye pair, 0.6 centre crop of 1280x720",
    ),
    "wrist_left": dict(
        topics=("hand_left",),
        model="pinhole",
        published=(1280, 720),
        note="Waveshare IMX335 5MP USB",
    ),
    "wrist_right": dict(
        topics=("hand_right",),
        model="pinhole",
        published=(1280, 720),
        note="Waveshare IMX335 5MP USB",
    ),
}

# Boards that exist in assets/calib/, plus the 8x6 used for the teleop
# episodes. Tried in order at startup so a wrong --board is not a dead run.
KNOWN_BOARDS = ((9, 6), (8, 6), (8, 5), (10, 6))

ZONE_NAMES = ("top-left", "top-centre", "top-right",
              "middle-left", "CENTRE", "middle-right",
              "bottom-left", "bottom-centre", "bottom-right")
TILT_EDGES = (0.0, 15.0, 35.0, 90.0)
TILT_NAMES = ("flat on to the lens", "tilted about 25 deg",
              "tilted hard, about 45 deg")
DIST_NAMES = ("FAR - small in the frame", "CLOSE - filling a third of it")

# Acceptance. None of these is a contrast threshold; see the module docstring.
MIN_AREA = 0.05          # board hull as a fraction of frame; below this the
                         # corners carry too little leverage to say anything
NEAR_AREA = 0.16         # near/far split
MAX_CLIP_HI = 0.02       # blown highlights inside the board = glare
MAX_CLIP_LO = 0.05       # crushed blacks
STILL_PX = 1.5           # per-corner motion between frames to count as still.
                         # Frames reaching the gate are now adjacent in time
                         # (see Stream.recv_latest), so this is a real hand
                         # tremor tolerance rather than a queue artefact.
HOLD_FRAMES = 2
DISTINCT_PX = 15.0       # a view closer than this to one already taken adds
                         # no new constraint

# When the scoreboard is considered full.
DEFAULTS = dict(zone_min=3, tilt_min=6, dist_min=8, total_min=30,
                tilt_std_min=12.0)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def make_objp(board, square):
    bw, bh = board
    objp = np.zeros((bw * bh, 3), np.float32)
    objp[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * square
    return objp


def slug(topic: str) -> str:
    return topic.replace("/", "_")


def py(o):
    """numpy -> plain python, so yaml.safe_dump does not choke."""
    if isinstance(o, dict):
        return {k: py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    return o


def detect(cv2, gray, board):
    """Find the board, then try again on a contrast-equalised copy.

    CLAHE is what makes ordinary room light workable. A sheet lit by one
    ceiling lamp has a brightness gradient across it; a global threshold
    misses the dim end, and the detector returns nothing at all. Equalising
    locally removes the gradient without touching corner geometry, which is
    the thing being measured.
    """
    from calibrate_cameras import detect_board
    found, corners = detect_board(cv2, gray, board)
    if found:
        return True, corners, False
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    found, corners = detect_board(cv2, eq, board)
    return found, corners, found


def frame_metrics(cv2, gray, corners):
    """Exposure, sharpness and geometry for one detection.

    Contrast is reported but never gated on - it is measured as a robust
    5th-to-95th-percentile Michelson so a clipped white does not flatter it,
    and sharpness is divided by contrast squared so that a correctly exposed
    dim board is not mistaken for a blurred one.
    """
    h, w = gray.shape
    pts = np.asarray(corners, np.float32).reshape(-1, 2)

    mask = np.zeros(gray.shape, np.uint8)
    hull = cv2.convexHull(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    inside = gray[mask > 0]
    if inside.size < 50:
        return None
    lo, hi = np.percentile(inside, [5.0, 95.0])
    contrast = float((hi - lo) / max(hi + lo, 1e-6))
    clip_hi = float((inside >= 254).mean())
    clip_lo = float((inside <= 1).mean())

    x0, y0 = pts.min(0).astype(int)
    x1, y1 = pts.max(0).astype(int)
    roi = gray[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 100 else 0.0
    sharp = 1e4 * lap / max((hi - lo) ** 2, 1.0)

    area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) / (w * h)
    cen = (float(pts[:, 0].mean() / w), float(pts[:, 1].mean() / h))
    return dict(contrast=contrast, clip_hi=clip_hi, clip_lo=clip_lo,
                lap=lap, sharp=sharp, area=area, cen=cen,
                mean=float(inside.mean()), p5=float(lo), p95=float(hi))


def board_pose(cv2, pts, size, board, square):
    """Tilt of the board off the image plane, and its in-plane roll.

    Only the angle is used, and angles do not depend on the seed focal length
    strongly enough to matter for binning - nor on the square size at all.
    """
    objp = make_objp(board, square)
    w, h = size
    K = np.array([[0.8 * w, 0, w / 2.0], [0, 0.8 * w, h / 2.0], [0, 0, 1.0]],
                 float)
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    try:
        ok, rvec, _ = cv2.solvePnP(objp, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    tilt = math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))
    roll = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return tilt, roll


# --------------------------------------------------------------------------
# the stream
# --------------------------------------------------------------------------
class Stream:
    """The published ZMQ stream, always served newest-first.

    Board detection costs more than a frame interval, so a plain recv() loop
    falls behind: the socket backs up, every frame processed is seconds stale,
    the steering text describes where the board WAS, and two consecutively
    processed frames come from different poses - so the stillness test can
    never pass. recv_latest() drains whatever has queued and decodes only the
    survivor, which keeps the operator's feedback live. It also hands back one
    frame per topic captured at the SAME instant, which is what makes the
    second eye of a synchronous stereo pair usable.
    """

    def __init__(self, connect, topics, timeout_ms=1000):
        import zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.connect(connect)
        for t in topics:
            self.sock.setsockopt_string(zmq.SUBSCRIBE, t)
        self.timeout_ms = timeout_ms
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.arrivals = {t: [] for t in topics}
        self.headers = {}
        self._zmq = zmq

    def _recv_raw(self):
        """(topic, jpeg bytes) without paying for a decode."""
        try:
            parts = self.sock.recv_multipart()
        except self._zmq.Again:
            return None
        if len(parts) < 2:
            return None
        topic = parts[0].decode(errors="replace")
        if len(parts) >= 3:
            try:
                self.headers[topic] = json.loads(parts[1])
            except (ValueError, UnicodeDecodeError):
                pass
        self.arrivals.setdefault(topic, []).append(time.time())
        return topic, parts[-1]

    def recv_latest(self):
        """-> {topic: bgr image} for the newest frame of each topic."""
        import cv2
        first = self._recv_raw()          # one blocking read, so we never spin
        if first is None:
            return {}
        raw = {first[0]: first[1]}
        self.sock.setsockopt(self._zmq.RCVTIMEO, 0)
        try:
            while True:
                nxt = self._recv_raw()
                if nxt is None:
                    break
                raw[nxt[0]] = nxt[1]
        except self._zmq.Again:
            pass
        finally:
            self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        out = {}
        for topic, payload in raw.items():
            img = cv2.imdecode(np.frombuffer(payload, np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                out[topic] = img
        return out

    def fps(self, topic):
        t = np.asarray(self.arrivals.get(topic, []), float)
        if t.size < 8:
            return None
        dt = np.diff(t)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if dt.size else None


# --------------------------------------------------------------------------
# the coverage scoreboard
# --------------------------------------------------------------------------
class Coverage:
    """Tracks the space that determines whether the fit is conditioned.

    Counting views hides the failure mode that produced the numbers this
    script replaces: forty views all at the same tilt constrain no better than
    one. So the stopping rule is over zone, tilt and distance, not over count.
    """

    def __init__(self, req):
        self.req = req
        self.zones = np.zeros(9, int)
        self.tilts = np.zeros(3, int)
        self.dists = np.zeros(2, int)
        self.cells = {}
        self.tilt_vals = []

    @staticmethod
    def bin_of(metrics, tilt):
        cx, cy = metrics["cen"]
        zone = min(2, int(cy * 3)) * 3 + min(2, int(cx * 3))
        tb = 0
        for i in range(3):
            if TILT_EDGES[i] <= tilt < TILT_EDGES[i + 1]:
                tb = i
        db = 1 if metrics["area"] >= NEAR_AREA else 0
        return zone, tb, db

    def add(self, metrics, tilt):
        z, t, d = self.bin_of(metrics, tilt)
        self.zones[z] += 1
        self.tilts[t] += 1
        self.dists[d] += 1
        self.cells[(z, t, d)] = self.cells.get((z, t, d), 0) + 1
        self.tilt_vals.append(tilt)

    @property
    def total(self):
        return int(self.zones.sum())

    @property
    def tilt_std(self):
        return float(np.std(self.tilt_vals)) if len(self.tilt_vals) > 1 else 0.0

    def unmet(self):
        r, out = self.req, []
        for i, n in enumerate(self.zones):
            if n < r["zone_min"]:
                out.append((r["zone_min"] - n, "zone", i))
        for i, n in enumerate(self.tilts):
            if n < r["tilt_min"]:
                out.append((r["tilt_min"] - n, "tilt", i))
        for i, n in enumerate(self.dists):
            if n < r["dist_min"]:
                out.append((r["dist_min"] - n, "dist", i))
        if self.total < r["total_min"]:
            out.append((r["total_min"] - self.total, "total", None))
        if self.tilt_std < r["tilt_std_min"]:
            out.append((1, "tiltstd", None))
        return sorted(out, reverse=True)

    def done(self):
        return not self.unmet()

    def steer(self):
        """The single most useful thing to ask for next."""
        unmet = self.unmet()
        if not unmet:
            return None
        deficit, kind, key = unmet[0]
        # Name a concrete pose: the neediest zone, crossed with whichever tilt
        # and distance are themselves shortest, so one move fixes several.
        zone = key if kind == "zone" else int(np.argmin(self.zones))
        tilt = key if kind == "tilt" else int(np.argmin(self.tilts))
        dist = key if kind == "dist" else int(np.argmin(self.dists))
        if kind == "tiltstd":
            return ("vary the tilt MORE - every view so far is at a similar "
                    f"angle (spread {self.tilt_std:.0f} deg, need "
                    f"{self.req['tilt_std_min']:.0f})")
        return (f"{ZONE_NAMES[zone]} of the frame, {TILT_NAMES[tilt]}, "
                f"held {DIST_NAMES[dist]}   [{deficit} more]")

    def report(self):
        lines = [f"    zones    {' '.join(f'{n:2d}' for n in self.zones)}"
                 f"   (need {self.req['zone_min']} each)",
                 f"    tilt     {' '.join(f'{n:2d}' for n in self.tilts)}"
                 f"   flat / 25deg / 45deg  (need {self.req['tilt_min']} each)",
                 f"    distance {' '.join(f'{n:2d}' for n in self.dists)}"
                 f"   far / close  (need {self.req['dist_min']} each)",
                 f"    total {self.total}  tilt spread {self.tilt_std:.1f} deg"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# lighting probe
# --------------------------------------------------------------------------
def probe_lighting(cv2, stream, lead, board, seconds=8.0):
    """Grade the room, instead of asking for it to be changed.

    Reports what the exposure is actually doing and whether the board is being
    found. The only genuinely fatal lighting problem for a printed board is
    specular glare, which shows up as clipped highlights inside the board -
    not low contrast, which the detector handles and CLAHE rescues.
    """
    print(f"\n  Grading the lighting on {lead} for {seconds:.0f} s - "
          f"hold the board up and move it around.\n")
    t0, seen, tried, stats, clahe_saves = time.time(), 0, 0, [], 0
    while time.time() - t0 < seconds:
        img = stream.recv_latest().get(lead)
        if img is None:
            continue
        tried += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners, used_clahe = detect(cv2, gray, board)
        if not found:
            print("\r    board not visible" + " " * 50, end="", flush=True)
            continue
        m = frame_metrics(cv2, gray, corners)
        if m is None:
            continue
        seen += 1
        clahe_saves += int(used_clahe)
        stats.append(m)
        print(f"\r    found  contrast {m['contrast']:.2f}  glare "
              f"{100*m['clip_hi']:4.1f}%  sharp {m['sharp']:5.1f}  "
              f"fill {100*m['area']:4.1f}%" + " " * 8, end="", flush=True)
    print()
    if not stats:
        print(f"\n  Never saw a {board[0]}x{board[1]} board in {tried} frames.")
        print("  Either the board is not in this camera's view, or it is the "
              "wrong size.")
        return False
    det = 100.0 * seen / max(tried, 1)
    glare = float(np.median([s["clip_hi"] for s in stats]))
    worst = float(np.max([s["clip_hi"] for s in stats]))
    contrast = float(np.median([s["contrast"] for s in stats]))
    mean = float(np.median([s["mean"] for s in stats]))
    print(f"\n  detection rate   {det:.0f}% of frames")
    print(f"  board brightness {mean:.0f}/255   contrast {contrast:.2f}")
    print(f"  glare (clipped)  {100*glare:.1f}% median, {100*worst:.1f}% worst")
    if clahe_saves:
        print(f"  {clahe_saves} of {seen} detections needed contrast "
              f"equalisation - uneven light across the sheet, which is fine")
    if worst > MAX_CLIP_HI * 3:
        print("\n  GLARE. There is a specular reflection of a lamp on the "
              "paper.\n  Tilt the board away from the light, or move one step "
              "sideways.\n  This - not brightness - is the thing that would "
              "spoil the corners.")
    elif det < 25:
        print("\n  The board is often missed. Hold it more square to the "
              "camera and\n  closer while you start; the sweep will ask for "
              "the hard angles later.")
    else:
        print("\n  LIGHTING IS FINE. No need to touch the camera exposure.")
    return True


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------
def capture(cv2, stream, cfg, board, square, outdir, req, existing=None):
    """Steer the operator until the coverage space is full.

    Every detection from every subscribed topic is written out, not only the
    ones that close a gap: they cost nothing at capture time and let --refit
    reach a different answer without another session with the board.
    """
    topics = cfg["topics"]
    lead = topics[0]
    cov = Coverage(req)
    recs = {t: [] for t in topics}
    accepted = {t: [] for t in topics}
    sizes = {}
    counter = {t: 0 for t in topics}
    manifest = (outdir / "manifest.jsonl").open("a", encoding="utf-8")

    if existing:
        for t, rs in existing.items():
            recs.setdefault(t, []).extend(rs)
            counter[t] = max(counter.get(t, 0), len(rs))
            for r in rs:
                if not r.get("accepted"):
                    continue
                accepted.setdefault(t, []).append(
                    np.array(r["corners"], np.float32))
                sizes[t] = tuple(r["size"])
                if t == lead:
                    cov.add(r["metrics"], r["tilt"])
        print(f"  resumed with {cov.total} views already on {lead}\n")

    print(f"\n  Calibrating {', '.join(topics)}")
    print(f"  Board {board[0]}x{board[1]} inner corners")
    print("  Hold the board where this camera can see it. The line below "
          "tells you\n  where to put it next. Ctrl-C stops early and still "
          "fits what you have.")
    print("  " + "-" * 68)

    def assess(topic, img):
        """Detect and grade one frame. -> (rec, pts, metrics, tilt) or None."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sizes[topic] = gray.shape[::-1]
        found, corners, used_clahe = detect(cv2, gray, board)
        if not found:
            return None
        m = frame_metrics(cv2, gray, corners)
        if m is None:
            return None
        pose = board_pose(cv2, corners, sizes[topic], board, square)
        if pose is None:
            return None
        tilt, roll = pose
        pts = np.asarray(corners, np.float32).reshape(-1, 2)
        rec = dict(topic=topic, file=None, size=list(sizes[topic]),
                   corners=pts.tolist(), tilt=tilt, roll=roll, metrics=py(m),
                   clahe=bool(used_clahe), stamp=time.time(), accepted=False)
        return rec, pts, m, tilt

    def why_not(topic, pts, m):
        """The single most useful thing to say about a frame we will not take."""
        if m["clip_hi"] > MAX_CLIP_HI:
            return (f"glare on the board ({100*m['clip_hi']:.0f}% of it is "
                    f"blown out) - tilt it away from the light")
        if m["clip_lo"] > MAX_CLIP_LO:
            return "the board is in deep shadow - turn it toward the room"
        if m["area"] < MIN_AREA:
            return (f"too far away ({100*m['area']:.0f}% of the frame) - "
                    f"bring it closer")
        if any(np.abs(pts - q).mean() < DISTINCT_PX for q in accepted[topic]):
            return "too close to a view already taken - move it further"
        return None

    def keep(rec, pts, topic, img):
        """Accept a view: write the frame, the corners, and the manifest line."""
        counter[topic] += 1
        rec["file"] = f"{slug(topic)}/{counter[topic]:04d}.png"
        (outdir / slug(topic)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outdir / rec["file"]), img)
        rec["accepted"] = True
        accepted[topic].append(pts)
        recs[topic].append(rec)
        manifest.write(json.dumps(py(rec)) + "\n")
        manifest.flush()

    hold, last_pts, last_msg, t_msg = 0, None, "", 0.0
    try:
        while not cov.done():
            batch = stream.recv_latest()
            img = batch.get(lead)
            if img is None:
                print("\r    waiting for frames..." + " " * 50,
                      end="", flush=True)
                continue

            got = assess(lead, img)
            if got is None:
                hold, last_pts = 0, None
                print("\r    board not visible" + " " * 50, end="", flush=True)
                continue
            rec, pts, m, tilt = got

            reason = why_not(lead, pts, m)
            if reason is None:
                still = (last_pts is not None and last_pts.shape == pts.shape
                         and float(np.abs(pts - last_pts).mean()) < STILL_PX)
                if not still:
                    reason = "hold it still"
            last_pts = pts

            if reason is not None:
                hold = 0
                # Corners of a rejected frame still cost nothing to keep, and
                # --refit can reach a different answer with them. The image is
                # not kept: at ten detections a second that is gigabytes.
                manifest.write(json.dumps(py(rec)) + "\n")
                if reason != last_msg or time.time() - t_msg > 0.6:
                    print(f"\r    {reason}" + " " * 25, end="", flush=True)
                    last_msg, t_msg = reason, time.time()
                continue

            hold += 1
            if hold < HOLD_FRAMES:
                print(f"\r    good - hold still {HOLD_FRAMES - hold}..."
                      + " " * 30, end="", flush=True)
                manifest.write(json.dumps(py(rec)) + "\n")
                continue
            hold = 0

            keep(rec, pts, lead, img)
            cov.add(m, tilt)

            # The other eye, from the SAME drained batch - so it is the same
            # instant, not whatever arrived while this one was being detected.
            # Detecting it only on an accepted frame also halves the work,
            # which is what makes the loop keep up with the stream at all.
            for other in topics[1:]:
                oimg = batch.get(other)
                if oimg is None:
                    continue
                o = assess(other, oimg)
                if o is None:
                    continue
                orec, opts, om, _ = o
                if why_not(other, opts, om) is None:
                    keep(orec, opts, other, oimg)
                else:
                    manifest.write(json.dumps(py(orec)) + "\n")

            extra = "".join(f"  {slug(t)} {len(accepted[t])}"
                            for t in topics[1:])
            print(f"\r    TAKEN {cov.total}  tilt {tilt:4.0f} deg  "
                  f"fill {100*m['area']:4.1f}%{extra}" + " " * 12, flush=True)
            nxt = cov.steer()
            if nxt:
                print(f"    -> next: {nxt}")
    except KeyboardInterrupt:
        print("\n\n  stopped early - fitting what was captured")
    finally:
        manifest.close()

    all_recs = {t: [r for r in recs[t]] for t in topics}
    # Anything written to the manifest but not accepted is still available to
    # --refit; only the accepted ones are returned for the immediate fit.
    print("\n  " + "-" * 68)
    print("  coverage reached:")
    print(cov.report())
    return all_recs, sizes, cov


# --------------------------------------------------------------------------
# fitting and validation
# --------------------------------------------------------------------------
def fit_model(cv2, corners, size, board, square, model, free_k3, max_drop=12):
    """Calibrate, releasing the third radial term only when it is identifiable.

    calibrate_cameras.fit frees k3 as soon as there are 20 views, and view
    count is the wrong proxy: forty views crowded into the middle of the frame
    still cannot identify a third radial term, and it runs away absorbing
    noise instead. On a deliberately zone-starved synthetic set that lands on
    k3 = -2.5 while the residual stays at 0.109 px - which is precisely how
    the old head.yaml reached D = [-0.3, 2.15, -6.78, 6.98] and still looked
    healthy. What makes k3 identifiable is coverage of the frame, so coverage
    is what gates it.

    The degenerate-view filtering and drop-and-retry are the upstream ones;
    only the flag decision differs.
    """
    from calibrate_cameras import _is_sane, _degeneracy_rank, _fit_once, seed_K
    sane = [c for c in corners if _is_sane(cv2, c, size)]
    if len(sane) >= 8:
        corners = sane
    order = _degeneracy_rank(cv2, corners, size, board, square)
    keep = list(range(len(corners)))

    def once(sub):
        if model == "fisheye":
            return _fit_once(cv2, sub, size, board, square, model)
        objp = make_objp(board, square)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if not free_k3:
            flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        rms, K, D, *_ = cv2.calibrateCamera(
            [objp for _ in sub], sub, size, seed_K(size, "pinhole"), None,
            flags=flags)
        return float(rms), np.asarray(K), np.asarray(D).reshape(-1)

    for _ in range(max_drop + 1):
        try:
            return once([corners[i] for i in keep])
        except cv2.error:
            drop = next((i for i in order if i in keep), None)
            if drop is None or len(keep) <= 8:
                raise
            keep.remove(drop)
    raise RuntimeError("calibration kept failing after dropping views")


def k3_is_identifiable(cov, n_views):
    """Whether the views actually span enough of the frame to support k3."""
    if cov is None:
        return n_views >= 20                    # upstream's rule, as fallback
    return bool((cov.zones >= 2).all() and cov.total >= 25)


def reproj_error(cv2, pts, K, D, board, square, model):
    """Reprojection error of ONE view under an already-fitted K, D.

    Used on views the fit never saw, which is the only honest test: a fit that
    has absorbed pose error into its distortion terms - the runaway
    D = [-0.3, 2.15, -6.78, 6.98] in the old head.yaml - looks fine in-sample
    and falls apart here.
    """
    objp = make_objp(board, square)
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    try:
        if model == "fisheye":
            Df = np.asarray(D, float).reshape(4, 1)
            und = cv2.fisheye.undistortPoints(
                pts.reshape(1, -1, 2).astype(np.float64), K, Df)
            und = und.reshape(-1, 1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, und, np.eye(3), None,
                                          flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                return None
            proj, _ = cv2.fisheye.projectPoints(
                objp.reshape(1, -1, 3).astype(np.float64), rvec, tvec, K, Df)
        else:
            Dp = np.asarray(D, float).reshape(-1, 1)
            ok, rvec, tvec = cv2.solvePnP(objp, pts, K, Dp,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, Dp)
    except cv2.error:
        return None
    proj = np.asarray(proj, float).reshape(-1, 2)
    d = proj - pts.reshape(-1, 2)
    return float(np.sqrt((d ** 2).sum(1).mean()))


def kfold_error(cv2, corners, size, board, square, model, free_k3, k=5):
    n = len(corners)
    if n < 15:
        return None
    idx = np.arange(n)
    fold = idx % k
    errs = []
    for f in range(k):
        tr = [corners[i] for i in idx[fold != f]]
        te = [corners[i] for i in idx[fold == f]]
        if len(tr) < 10 or not te:
            continue
        try:
            _, K, D = fit_model(cv2, tr, size, board, square, model, free_k3)
        except Exception:
            continue
        for c in te:
            e = reproj_error(cv2, c, K, D, board, square, model)
            if e is not None and np.isfinite(e):
                errs.append(e)
    return float(np.mean(errs)) if errs else None


def bootstrap_ci(cv2, corners, size, board, square, model, free_k3,
                 n=40, seed=0):
    """How much the answer moves when the views are resampled.

    This is the number that turns `is this trustworthy` into something you can
    read off. A calibration whose fx moves 15% under resampling is not a
    measurement, whatever its residual says.
    """
    if n <= 0 or len(corners) < 12:
        return None
    rng = np.random.default_rng(seed)
    got = {"fx": [], "fy": [], "cx": [], "cy": []}
    for _ in range(n):
        pick = rng.integers(0, len(corners), len(corners))
        try:
            _, K, _ = fit_model(cv2, [corners[i] for i in pick], size,
                                board, square, model, free_k3)
        except Exception:
            continue
        got["fx"].append(K[0, 0]); got["fy"].append(K[1, 1])
        got["cx"].append(K[0, 2]); got["cy"].append(K[1, 2])
    if len(got["fx"]) < max(8, n // 4):
        return None
    out = {}
    for key, vals in got.items():
        v = np.asarray(vals, float)
        out[key] = dict(mean=float(v.mean()), std=float(v.std()),
                        lo=float(np.percentile(v, 2.5)),
                        hi=float(np.percentile(v, 97.5)),
                        n=int(v.size))
    return out


def select_views(cv2, recs, max_views):
    """Pick a well-spread subset, best-first within each cell.

    Feeding the solver everything is not free: near-duplicates add weight
    without adding constraint, and pull the answer toward whatever region the
    board happened to linger in. Round-robin over the coverage cells spreads
    the weight evenly instead.
    """
    if len(recs) <= max_views:
        return recs
    cells = {}
    for r in recs:
        key = Coverage.bin_of(r["metrics"], r["tilt"])
        cells.setdefault(key, []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: -r["metrics"]["sharp"])
    out, keys = [], sorted(cells)
    i = 0
    while len(out) < max_views:
        progressed = False
        for k in keys:
            if i < len(cells[k]):
                out.append(cells[k][i])
                progressed = True
                if len(out) >= max_views:
                    break
        if not progressed:
            break
        i += 1
    return out


def expected_aspect(size):
    """fy/fx that the publish chain forces, for a frame of this shape.

    Every stream leaves the server 16:9 with square pixels. Resizing (W,H) to
    (w,h) scales fx by w/W and fy by h/H, so fy/fx = (H/h)/(W/w) reduces to
    source_aspect / this_frame's_aspect - 1.0 for an untouched 16:9 frame,
    exactly 4/3 for one already squashed to 640x480.
    """
    w, h = size
    return SOURCE_ASPECT / (w / h)


def run_checks(cv2, rms, K, D, corners, size, board, square, model,
               cov, boot, holdout, req):
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    exp_a = expected_aspect(size)
    got_a = fy / fx
    checks = []

    def add(name, value, ok, want):
        checks.append(dict(check=name, value=value,
                           verdict="PASS" if ok else "FAIL", want=want))

    add("rms reprojection", f"{rms:.3f} px", rms < 0.5, "< 0.5 px")
    if holdout is None:
        checks.append(dict(check="held-out reprojection", value="not run",
                           verdict="SKIP", want="< 1.5x rms"))
    else:
        add("held-out reprojection (5-fold)", f"{holdout:.3f} px",
            holdout < max(1.5 * rms, 0.6), "< 1.5x rms")
    if boot is None:
        checks.append(dict(check="bootstrap CI", value="not run",
                           verdict="SKIP", want="< 2%"))
    else:
        for k, v in (("fx", fx), ("fy", fy)):
            b = boot[k]
            half = 0.5 * (b["hi"] - b["lo"])
            add(f"{k} bootstrap 95% CI", f"{v:.1f} +- {half:.1f} px "
                f"({100*half/max(v,1e-9):.1f}%)", half / max(v, 1e-9) < 0.02,
                "< 2%")
    add(f"fy/fx vs predicted {exp_a:.4f}",
        f"{got_a:.4f} ({100*(got_a/exp_a - 1):+.1f}%)",
        abs(got_a / exp_a - 1.0) < 0.03, "within 3%")
    dx, dy = cx - w / 2.0, cy - h / 2.0
    add("principal point offset",
        f"({dx:+.1f}, {dy:+.1f}) px", abs(dx) < 0.15 * w / 2
        and abs(dy) < 0.15 * h / 2, "< 15% of half-size")
    if cov is not None:
        add("tilt spread", f"{np.mean(cov.tilt_vals):.1f} +- "
            f"{cov.tilt_std:.1f} deg", cov.tilt_std >= req["tilt_std_min"],
            f"std >= {req['tilt_std_min']:.0f} deg")
        add("zone coverage", f"{int((cov.zones >= req['zone_min']).sum())} of 9",
            bool((cov.zones >= req["zone_min"]).all()),
            f"9 of 9 with >= {req['zone_min']}")
    # A real lens of this kind lands well inside +-1 on every coefficient.
    # Larger means the terms are absorbing pose error rather than describing
    # the optics, and the model will not extrapolate to pixels the board never
    # visited - which is most of them. The old head.yaml sat at 6.98 here.
    dmax = float(np.max(np.abs(np.asarray(D, float).ravel())))
    add("distortion magnitude", f"max|D| = {dmax:.3f}", dmax < 1.0, "< 1.0")
    add("views used", f"{len(corners)}", len(corners) >= 25, ">= 25")
    return checks


FIX_HINTS = {
    "rms reprojection":
        "corners are noisy - most often a bowed sheet. Tape the board to "
        "stiff card and rerun.",
    "held-out reprojection (5-fold)":
        "the distortion terms are absorbing pose error rather than modelling "
        "the lens. Capture more views at DIFFERENT tilts, not more views.",
    "fx bootstrap 95% CI":
        "focal length is not pinned down. It needs board views at clearly "
        "different DISTANCES - near and far - to separate f from range.",
    "fy bootstrap 95% CI":
        "as fx: more separation between near and far views.",
    "principal point offset":
        "the board never reached the edges. Work the corners and the top and "
        "bottom of the frame.",
    "tilt spread":
        "every view is at a similar angle, which is exactly what made the old "
        "fit wrong. Slant the board hard, both ways, on both axes.",
    "zone coverage":
        "some regions of the image were never covered; distortion there is "
        "extrapolated, not measured.",
    "distortion magnitude":
        "the distortion terms have run away absorbing error. Almost always "
        "the same root cause as a failed zone or tilt check above - fix "
        "those and this follows.",
}


def calibrate_topic(cv2, topic, recs, size, board, square, model_pref,
                    cov, args):
    """Fit, validate, and - for the head - decide which lens model to trust."""
    used = select_views(cv2, recs, args.max_views)
    corners = [np.asarray(r["corners"], np.float32).reshape(-1, 1, 2)
               for r in used]
    if len(corners) < 8:
        print(f"\n  {topic}: only {len(corners)} usable views - cannot fit")
        return None

    free_k3 = k3_is_identifiable(cov, len(corners))
    models = (["pinhole", "fisheye"] if model_pref == "auto" else [model_pref])
    tried = {}
    for m in models:
        try:
            rms, K, D = fit_model(cv2, corners, size, board, square, m,
                                  free_k3)
        except Exception as exc:
            print(f"  {topic}: {m} fit failed ({exc})")
            continue
        ho = kfold_error(cv2, corners, size, board, square, m, free_k3,
                         k=args.folds)
        tried[m] = dict(rms=rms, K=K, D=D, holdout=ho)

    if not tried:
        print(f"\n  {topic}: no model could be fitted")
        return None

    # Choose on held-out error, not in-sample residual. The published head
    # stream is a 0.6 centre crop, which throws away the annulus where the
    # fisheye terms live, so a pinhole usually generalises better there - the
    # old fisheye fit's D of +-7 is what over-parameterising looks like.
    def rank(m):
        d = tried[m]
        return (d["holdout"] if d["holdout"] is not None else d["rms"] * 2)

    model = min(tried, key=rank)
    # When the two are within a few percent they are fitting the same thing and
    # the winner is noise, so take the one with fewer free parameters. That is
    # not merely tidiness: the published head stream is a 0.6 centre crop, which
    # discards the annulus where the fisheye terms are identifiable, so an
    # unconstrained fisheye there has four coefficients with nothing to pin them
    # down. D = [-0.3, 2.15, -6.78, 6.98] in the old head.yaml is what that
    # looks like when it goes wrong.
    if len(tried) > 1 and "pinhole" in tried and model != "pinhole":
        if rank("pinhole") <= rank(model) * 1.05:
            model = "pinhole"
    best = tried[model]
    rms, K, D, holdout = best["rms"], best["K"], best["D"], best["holdout"]

    if len(tried) > 1:
        print(f"\n  {topic}: lens model chosen on held-out error"
              f" (ties go to the simpler model)")
        for m, d in tried.items():
            mark = "  <-- used" if m == model else ""
            ho = f"{d['holdout']:.4f}" if d["holdout"] is not None else "   n/a"
            print(f"      {m:8s} rms {d['rms']:.4f} px   held-out {ho} px{mark}")

    print(f"  {topic}: bootstrapping ({args.bootstrap} resamples)...",
          flush=True)
    boot = bootstrap_ci(cv2, corners, size, board, square, model, free_k3,
                        n=args.bootstrap)

    if not free_k3:
        print(f"  {topic}: k3 and tangential terms held at zero - the views do "
              f"not span enough of the frame to identify them")
    checks = run_checks(cv2, rms, K, D, corners, size, board, square, model,
                        cov, boot, holdout, args.req)
    return dict(topic=topic, model=model, rms=rms, K=K, D=D, size=size,
                n_views=len(corners), n_candidates=len(recs), checks=checks,
                bootstrap=boot, holdout=holdout,
                alternatives={m: dict(rms=d["rms"], holdout=d["holdout"])
                              for m, d in tried.items()})


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def sim_block(res, published, fps):
    """Everything needed to point a MuJoCo camera at the same scene.

    The anisotropic squash to 640x480 preserves BOTH field angles - it changes
    the pixel aspect, not the field of view. What it means is that a MuJoCo
    camera, which has square pixels and one fovy, cannot render 640x480 with
    both angles right: it has to render at the published size and be resized
    afterwards, exactly as vla_camera_server.py already does. Both resolutions
    are therefore written out together, so the fovy cannot be pasted against
    the wrong one.
    """
    K, (w, h) = res["K"], res["size"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = published
    ow, oh = VLA_OUT
    fovx = 2 * math.degrees(math.atan(0.5 * w / fx))
    fovy = 2 * math.degrees(math.atan(0.5 * h / fy))
    f_pub = 0.5 * (fx * W / w + fy * H / h)
    return dict(
        published_resolution=[W, H],
        output_resolution=[ow, oh],
        fovy_deg=fovy,
        fovx_deg=fovx,
        f_px_published=f_pub,
        fx_out=fx * ow / w, fy_out=fy * oh / h,
        cx_out=cx * ow / w, cy_out=cy * oh / h,
        measured_fps=fps,
        mujoco=dict(
            fovy=round(fovy, 3),
            render_resolution=[W, H],
            then_resize_to=[ow, oh],
            note="MuJoCo has square pixels; render at render_resolution and "
                 "cv2.resize to then_resize_to, the way the real client does.",
        ),
    )


def write_yaml(res, cfg, board, square, fps, outdir, capture_dir):
    import yaml
    topic = res["topic"]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{slug(topic)}.yaml"
    if path.exists():
        keep = outdir / "superseded"
        keep.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.rename(keep / f"{slug(topic)}_{stamp}.yaml")
        print(f"     previous {path.name} moved to superseded/")
    doc = dict(
        model=res["model"],
        image_width=int(res["size"][0]), image_height=int(res["size"][1]),
        K=py(res["K"].tolist()), D=py(np.asarray(res["D"]).ravel().tolist()),
        rms_px=py(res["rms"]),
        holdout_px=py(res["holdout"]),
        n_views=res["n_views"], n_candidates=res["n_candidates"],
        board=f"{board[0]}x{board[1]}", square_m=square,
        source="live zmq stream, coverage-driven sweep",
        captured=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        capture_dir=str(capture_dir),
        note=cfg["note"],
        sim_camera=py(sim_block(res, cfg["published"], fps)),
        validation=py(dict(checks=res["checks"], bootstrap=res["bootstrap"],
                           model_comparison=res["alternatives"])),
    )
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def print_result(res, cfg, fps):
    K = res["K"]
    w, h = res["size"]
    print(f"\n  {'=' * 68}")
    print(f"  {res['topic']}   {w}x{h}   model {res['model']}   "
          f"{res['n_views']} of {res['n_candidates']} views")
    print(f"  {'=' * 68}")
    print(f"    fx {K[0,0]:8.2f}   fy {K[1,1]:8.2f}   "
          f"cx {K[0,2]:7.2f}   cy {K[1,2]:7.2f}")
    print(f"    D  {np.round(np.asarray(res['D']).ravel(), 5).tolist()}")
    print()
    print(f"    {'CHECK':<34}{'VALUE':<24}VERDICT")
    print(f"    {'-' * 66}")
    for c in res["checks"]:
        print(f"    {c['check']:<34}{str(c['value']):<24}{c['verdict']}")
    fails = [c for c in res["checks"] if c["verdict"] == "FAIL"]
    sim = sim_block(res, cfg["published"], fps)
    print()
    print(f"    FOR THE SIM   fovy {sim['fovy_deg']:.3f} deg   "
          f"fovx {sim['fovx_deg']:.3f} deg")
    print(f"                  render {sim['published_resolution'][0]}x"
          f"{sim['published_resolution'][1]}, resize to "
          f"{sim['output_resolution'][0]}x{sim['output_resolution'][1]}")
    print(f"                  at 640x480: fx {sim['fx_out']:.1f}  "
          f"fy {sim['fy_out']:.1f}  cx {sim['cx_out']:.1f}  "
          f"cy {sim['cy_out']:.1f}")
    if fps:
        print(f"                  measured {fps:.1f} fps")
    if fails:
        print(f"\n    {len(fails)} CHECK(S) FAILED - do not put these numbers "
              f"in the sim yet:")
        for c in fails:
            hint = FIX_HINTS.get(c["check"], "")
            print(f"      - {c['check']}: {c['value']}  (want {c['want']})")
            if hint:
                print(f"        {hint}")
        print("\n    Every frame is on disk, so add views with --resume "
              "<capture_dir>\n    and refit, rather than starting over.")
    else:
        print("\n    ALL CHECKS PASSED - these are safe to copy into the sim.")
    return not fails


# --------------------------------------------------------------------------
# board identification
# --------------------------------------------------------------------------
def identify_board(cv2, stream, lead, want, seconds=10.0):
    """Work out which printed board is being held, instead of failing on it."""
    if want != "auto":
        return tuple(int(v) for v in want.lower().split("x"))
    print(f"\n  Looking for a checkerboard on {lead} - hold your print up "
          f"square to the camera.")
    t0 = time.time()
    while time.time() - t0 < seconds:
        img = stream.recv_latest().get(lead)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for b in KNOWN_BOARDS:
            found, _, _ = detect(cv2, gray, b)
            if found:
                print(f"  found a {b[0]}x{b[1]} board (inner corners)\n")
                return b
        print("\r    looking..." + " " * 40, end="", flush=True)
    print()
    raise SystemExit(
        "  Could not identify the board. Tried "
        + ", ".join(f"{a}x{b}" for a, b in KNOWN_BOARDS)
        + ".\n  Pass it explicitly, e.g. --board 9x6 (INNER corners: a 10x7 "
          "grid of\n  squares has 9x6 inner corners).")


def load_manifest(path: Path):
    recs = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.setdefault(r["topic"], []).append(r)
    return recs


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=sorted(CAMERAS),
                   help="which physical camera to calibrate")
    p.add_argument("--connect", default=os.environ.get("RS_ROBOT_CAMERA", "tcp://192.168.123.164:5555"),
                   help="the robot's published camera stream")
    p.add_argument("--lead", default=None,
                   help="which topic steers the capture, e.g. stereo/left. "
                        "Defaults to the camera's first topic")
    p.add_argument("--board", default="auto",
                   help="INNER corners, e.g. 9x6. 'auto' identifies the print")
    p.add_argument("--square", type=float, default=0.025,
                   help="printed square size in metres. Does NOT affect "
                        "fx/fy/cx/cy - scaling the target scales only the "
                        "board's distance - so it matters for board pose, not "
                        "for the intrinsics you are copying into the sim")
    p.add_argument("--model", default=None,
                   choices=["auto", "pinhole", "fisheye"],
                   help="lens model; 'auto' fits both and keeps whichever "
                        "generalises better on held-out views")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "calib")
    p.add_argument("--probe", action="store_true",
                   help="grade the lighting and the print, then exit")
    p.add_argument("--no-probe", action="store_true",
                   help="skip the lighting probe and start capturing")
    p.add_argument("--refit", type=Path, default=None,
                   help="re-fit from a previous capture directory, no camera "
                        "needed")
    p.add_argument("--resume", type=Path, default=None,
                   help="add more views to a previous capture directory")
    p.add_argument("--max-views", type=int, default=45,
                   help="cap on views fed to the solver; the rest stay on "
                        "disk for --refit")
    p.add_argument("--bootstrap", type=int, default=40,
                   help="resamples for the confidence intervals; 0 skips")
    p.add_argument("--folds", type=int, default=5)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args()
    a.req = {k: getattr(a, k) for k in DEFAULTS}

    import cv2

    # ---- offline re-fit --------------------------------------------------
    if a.refit is not None:
        man = a.refit / "manifest.jsonl"
        if not man.exists():
            raise SystemExit(f"no manifest.jsonl in {a.refit}")
        meta = json.loads((a.refit / "session.json").read_text()) \
            if (a.refit / "session.json").exists() else {}
        camera = a.camera or meta.get("camera")
        if camera not in CAMERAS:
            raise SystemExit("cannot tell which camera this capture is from; "
                             "pass --camera")
        cfg = CAMERAS[camera]
        board = tuple(meta.get("board", (9, 6))) if a.board == "auto" \
            else tuple(int(v) for v in a.board.lower().split("x"))
        square = meta.get("square", a.square)
        fps = meta.get("fps", {})
        recs = load_manifest(man)
        print(f"\n  Re-fitting {camera} from {a.refit}")
        print(f"  board {board[0]}x{board[1]}, "
              f"{sum(len(v) for v in recs.values())} detections on disk\n")
        ok_all = True
        for topic in cfg["topics"]:
            rs = [r for r in recs.get(topic, []) if r.get("accepted")]
            if len(rs) < 8:
                rs = recs.get(topic, [])
            if len(rs) < 8:
                print(f"  {topic}: {len(rs)} detections - too few")
                continue
            size = tuple(rs[0]["size"])
            cov = Coverage(a.req)
            for r in rs:
                cov.add(r["metrics"], r["tilt"])
            res = calibrate_topic(cv2, topic, rs, size, board, square,
                                  a.model or cfg["model"], cov, a)
            if res is None:
                continue
            ok_all &= print_result(res, cfg, fps.get(topic))
            path = write_yaml(res, cfg, board, square, fps.get(topic),
                              a.out, a.refit)
            print(f"    -> {path}")
        return 0 if ok_all else 2

    if not a.camera:
        p.error("--camera is required (or --refit <dir>)")
    cfg = dict(CAMERAS[a.camera])
    if a.lead:
        if a.lead not in cfg["topics"]:
            raise SystemExit(f"--lead {a.lead} is not one of "
                             f"{', '.join(cfg['topics'])}")
        cfg["topics"] = (a.lead,) + tuple(t for t in cfg["topics"]
                                          if t != a.lead)
    stream = Stream(a.connect, cfg["topics"])
    lead = cfg["topics"][0]
    print(f"\n  {a.camera}: {cfg['note']}")
    print(f"  subscribed to {', '.join(cfg['topics'])} on {a.connect}")

    board = identify_board(cv2, stream, lead, a.board)

    if a.probe:
        return 0 if probe_lighting(cv2, stream, lead, board) else 1
    if not a.no_probe:
        if not probe_lighting(cv2, stream, lead, board, seconds=8.0):
            return 1

    if a.resume is not None:
        capture_dir = a.resume
        existing = load_manifest(capture_dir / "manifest.jsonl") \
            if (capture_dir / "manifest.jsonl").exists() else {}
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = a.out / f"capture_{a.camera}_{stamp}"
        existing = {}
    capture_dir.mkdir(parents=True, exist_ok=True)

    recs, sizes, cov = capture(cv2, stream, cfg, board, a.square,
                               capture_dir, a.req, existing)
    fps = {t: stream.fps(t) for t in cfg["topics"]}
    (capture_dir / "session.json").write_text(json.dumps(py(dict(
        camera=a.camera, board=list(board), square=a.square,
        connect=a.connect, fps=fps,
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))), indent=2))

    ok_all = True
    for topic in cfg["topics"]:
        rs = recs.get(topic, [])
        if len(rs) < 8:
            print(f"\n  {topic}: {len(rs)} views - too few to fit."
                  + ("  Was the board ever in this eye?" if topic != lead
                     else ""))
            ok_all = False
            continue
        # The second eye of a stereo pair is steered indirectly, so build it
        # its own scoreboard rather than reporting the lead eye's coverage
        # against it - that is how the right eye went unnoticed last time.
        if topic == lead:
            tcov = cov
        else:
            tcov = Coverage(a.req)
            for r in rs:
                tcov.add(r["metrics"], r["tilt"])
        res = calibrate_topic(cv2, topic, rs, sizes[topic], board, a.square,
                              a.model or cfg["model"], tcov, a)
        if res is None:
            ok_all = False
            continue
        ok_all &= print_result(res, cfg, fps.get(topic))
        path = write_yaml(res, cfg, board, a.square, fps.get(topic),
                          a.out, capture_dir)
        print(f"    -> {path}")

    print(f"\n  Frames and manifest kept in {capture_dir}")
    print(f"  Re-fit without the board:  {Path(__file__).name} "
          f"--refit {capture_dir}")
    print(f"  Add more views:            {Path(__file__).name} "
          f"--camera {a.camera} --resume {capture_dir}\n")
    return 0 if ok_all else 2


if __name__ == "__main__":
    ensure_deps()
    sys.exit(main())
