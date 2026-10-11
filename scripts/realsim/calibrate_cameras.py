#!/usr/bin/env python3
"""Calibrate the G1's real cameras - the intrinsics no part of the stack has.

`camera.py` on the robot has always been able to undistort (`undistort: true`),
but it reads `calib/<topic>.yaml` and nothing ever wrote one, so every camera
has run raw and no sim camera could be matched to a real focal length. UVC
exposes no intrinsics, so the only way to get them is to measure them.

Calibration happens at exactly the V4L2 mode the real server opens, not at the
sensor's native mode. That matters: those capture modes are a sensor ROI
*plus* a downscale (measured in configs/robots/g1_29dof_with_hand.yaml), so a K
calibrated at native resolution cannot be rescaled onto them - the naive
`K[0] *= sx` in camera.py's Undistorter is only valid for a pure resize. Doing
it at the working mode sidesteps the transfer entirely.

    # on the robot, camera server stopped, board held in front of the lens
    calibrate_cameras.py --camera head       -o calib/
    calibrate_cameras.py --camera wrist_left -o calib/

    # or re-fit later from the frames it saved
    calibrate_cameras.py --camera head --images calib/frames_stereo_left -o calib/

Headless by design: it prints as it accepts each view, so there is nothing to
look at. Move the board between views - slanted, near, far, and into the
corners, which is where distortion is actually determined.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# The real capture modes, from the robot's own config.yaml. Calibrating at the
# mode the data is collected at is the whole point, so these are not options.
CAMERAS = {
    "head": dict(
        device="/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.1:1.0-video-index0",
        mode=(2560, 720), stereo=True, model="fisheye",
        topics=("stereo/left", "stereo/right"),
        # per-eye native width and the fitted image-circle radius, used to turn
        # the calibrated f into the lens's true maximum field angle
        native_eye_w=1600, downscale=0.8162, circle_radius=785.0),
    "wrist_left": dict(
        device="/dev/v4l/by-path/platform-3610000.xhci-usb-0:3:1.0-video-index0",
        mode=(1280, 720), stereo=False, model="pinhole",
        topics=("hand_left",)),
    "wrist_right": dict(
        device="/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.3:1.0-video-index0",
        mode=(1280, 720), stereo=False, model="pinhole",
        topics=("hand_right",)),
}


def detect_board(cv2, gray, board):
    """Locate the chessboard, robustly.

    findChessboardCornersSB copes with the low contrast, oblique views and
    wide-angle blur that defeat the classic detector - on a real board held at
    an angle in front of the fisheye, the classic one finds nothing at all
    while SB finds every corner. SB already returns subpixel-accurate corners,
    so cornerSubPix is only needed on the fallback path.
    """
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(
            gray, board, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return True, np.asarray(c, np.float32).reshape(-1, 1, 2)
    ok, c = cv2.findChessboardCorners(
        gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return False, None
    c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return True, np.asarray(c, np.float32).reshape(-1, 1, 2)


def split_eyes(frame, stereo: bool) -> dict:
    if not stereo:
        return {"": frame}
    half = frame.shape[1] // 2
    return {"left": frame[:, :half], "right": frame[:, half:]}


def grab_views(cv2, cfg, n_views: int, board, save_dir: Path | None):
    """Capture until `n_views` sufficiently distinct board detections per eye."""
    w, h = cfg["mode"]
    cap = cv2.VideoCapture(cfg["device"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {cfg['device']} - is the camera server still running?")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (got_w, got_h) != (w, h):
        raise SystemExit(f"asked for {w}x{h}, camera gave {got_w}x{got_h}")
    print(f"[calib] {cfg['device']} at {got_w}x{got_h}; move the board around", flush=True)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    views: dict[str, list] = {}
    size: dict[str, tuple] = {}
    last: dict[str, np.ndarray] = {}
    tries = 0
    while True:
        ok, frame = cap.read()
        tries += 1
        if not ok or frame is None:
            if tries > 3000:
                break
            continue
        eyes = split_eyes(frame, cfg["stereo"])
        for eye, img in eyes.items():
            views.setdefault(eye, [])
            if len(views[eye]) >= n_views:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = detect_board(cv2, gray, board)
            if not found:
                continue
            # Reject a view that barely moved, it adds no new constraint.
            if eye in last and np.abs(corners - last[eye]).mean() < 25.0:
                continue
            last[eye] = corners
            views[eye].append(corners)
            size[eye] = gray.shape[::-1]
            if save_dir is not None:
                d = save_dir / f"frames_{eye or 'mono'}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{len(views[eye]):03d}.png"), img)
            print(f"[calib]   {eye or 'mono'}: {len(views[eye])}/{n_views}", flush=True)
        if all(len(v) >= n_views for v in views.values()) and views:
            break
        if tries > 20000:
            print("[calib] giving up waiting for views", flush=True)
            break
    cap.release()
    return views, size


def load_views(cv2, img_dir: Path, board):
    """Re-detect the board in already-saved frames."""
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    out, size = [], None
    for p in sorted(img_dir.glob("*.png")):
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = detect_board(cv2, gray, board)
        if not found:
            print(f"[calib]   {p.name}: no board, skipped")
            continue
        out.append(corners)
        size = gray.shape[::-1]
    return out, size


def view_spread(cv2, corners, size, board, square):
    """How much the board's orientation actually varies across the views.

    Planar calibration is ill-conditioned when every view shows the board at
    the same tilt: focal length and distance trade off against each other and
    the solver can sit on its seed. Counting views hides that, so measure it.
    Returns (mean_tilt_deg, std_tilt_deg).
    """
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    w, h = size
    K = np.array([[w / 2, 0, w / 2], [0, w / 2, h / 2], [0, 0, 1]], float)
    tilts = []
    for c in corners:
        ok, rvec, _ = cv2.solvePnP(objp, np.asarray(c, np.float32).reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            tilts.append(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    if not tilts:
        return 0.0, 0.0
    return float(np.mean(tilts)), float(np.std(tilts))


def seed_K(size, model):
    """A physically sensible starting guess, so a marginal fit refines rather
    than diverges. The ELP's equidistant f is about 408 px in the published
    frame (500 px/rad on the sensor, times the 0.8162 capture downscale); the
    wrist lenses are wide but rectilinear, so ~0.7*width is close."""
    w, h = size
    fx = 408.0 * (w / 768.0) if model == "fisheye" else 0.7 * w
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]], float)


def _degeneracy_rank(cv2, corners, size, board, square):
    """Order views by how likely they are to break the solver, worst first.

    OpenCV's fisheye InitExtrinsics asserts (fabs(norm_u1) > 0) on a view whose
    board is close to edge-on, because the initial pose is recovered from a
    homography that has gone singular. One such view aborts the whole
    calibration, so they have to be identifiable before the solve.
    """
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    w, h = size
    K = np.array([[w / 2, 0, w / 2], [0, w / 2, h / 2], [0, 0, 1]], float)
    risk = []
    for i, c in enumerate(corners):
        pts = np.asarray(c, np.float32).reshape(-1, 2)
        ok, rvec, _ = cv2.solvePnP(objp, pts.reshape(-1, 1, 2), K, None,
                                   flags=cv2.SOLVEPNP_IPPE)
        tilt = 0.0
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
        area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) / (w * h)
        # edge-on (tilt -> 90) and tiny both make the homography ill-conditioned
        risk.append((tilt - 1000.0 * area, i))
    risk.sort(reverse=True)
    return [i for _, i in risk]


def _is_sane(cv2, c, size):
    """Reject a view whose corners cannot constrain a pose.

    Surviving the solver is not the bar - a near-collinear or vanishingly
    small detection does not throw, it just pulls the answer. In testing, one
    collinear view mixed into twenty-four good ones moved the recovered focal
    length by 3.4% while reporting a healthy sub-pixel residual, so these have
    to be dropped up front rather than tolerated.
    """
    pts = np.asarray(c, np.float32).reshape(-1, 2)
    w, h = size
    if np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]) < 0.004 * w * h:
        return False                                  # too small to say anything
    d = pts - pts.mean(0)
    sv = np.linalg.svd(d, compute_uv=False)
    return bool(sv[-1] / max(sv[0], 1e-9) > 0.05)     # not collapsed to a line


def fit(cv2, corners, size, board, square, model, max_drop=12):
    """Calibrate, excluding views that cannot support a pose.

    Two layers: obviously degenerate detections are filtered before the solve,
    and anything that still aborts the solver is dropped and retried, so a
    single bad view out of forty cannot lose the whole calibration.
    """
    sane = [c for c in corners if _is_sane(cv2, c, size)]
    if len(sane) >= 8:
        corners = sane
    order = _degeneracy_rank(cv2, corners, size, board, square)
    keep = list(range(len(corners)))
    for _ in range(max_drop + 1):
        try:
            return _fit_once(cv2, [corners[i] for i in keep], size, board,
                             square, model)
        except cv2.error:
            drop = next((i for i in order if i in keep), None)
            if drop is None or len(keep) <= 8:
                raise
            keep.remove(drop)
    raise RuntimeError("calibration kept failing after dropping views")


def _fit_once(cv2, corners, size, board, square, model):
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    if model == "fisheye":
        # cv2.fisheye.calibrate only accepts (1, N, 3) / (1, N, 2) float64; the
        # (N, 1, 2) float32 that findChessboardCorners returns makes it throw a
        # size-mismatch deep in arithm_op rather than anything legible.
        obj = [objp.astype(np.float64).reshape(1, -1, 3) for _ in corners]
        corners = [np.asarray(c, np.float64).reshape(1, -1, 2) for c in corners]
        K = seed_K(size, "fisheye"); D = np.zeros((4, 1))
        # OpenCV 4.x keeps these under cv2.fisheye, 5.x moved them to the top
        # level. The robot runs 4.13 and the workstation 5.0, so accept both.
        def flag(name):
            return getattr(cv2.fisheye, name, None) or getattr(cv2, name)
        flags = (flag("CALIB_RECOMPUTE_EXTRINSIC") | flag("CALIB_FIX_SKEW")
                 | flag("CALIB_USE_INTRINSIC_GUESS"))
        rms, K, D, *_ = cv2.fisheye.calibrate(
            obj, corners, size, K, D,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6))
    else:
        obj = [objp for _ in corners]
        # k3 and the tangential terms need a lot of well-spread views before
        # they are identifiable; with few, they soak up noise and send fy to
        # absurd values. Drop them until there is data to support them.
        pflags = cv2.CALIB_USE_INTRINSIC_GUESS
        if len(corners) < 20:
            pflags |= cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        rms, K, D, *_ = cv2.calibrateCamera(
            obj, corners, size, seed_K(size, "pinhole"), None, flags=pflags)
    return float(rms), np.asarray(K), np.asarray(D).reshape(-1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True, choices=sorted(CAMERAS))
    p.add_argument("--board", default="9x6",
                   help="INNER corner counts, not squares: a 10x7 board is 9x6")
    p.add_argument("--square", type=float, default=0.025,
                   help="printed square size in metres - measure it, do not trust the PDF")
    p.add_argument("--views", type=int, default=20)
    p.add_argument("--images", type=Path, default=None,
                   help="re-fit from a directory of saved frames instead of capturing")
    p.add_argument("-o", "--out", type=Path, default=Path("calib"))
    a = p.parse_args()

    import cv2
    import yaml

    cfg = CAMERAS[a.camera]
    bw, bh = (int(x) for x in a.board.lower().split("x"))
    board = (bw, bh)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.images is not None:
        corners, size = load_views(cv2, a.images, board)
        if not corners:
            raise SystemExit(f"no usable board views in {a.images}")
        views, sizes = {"": corners}, {"": size}
    else:
        views, sizes = grab_views(cv2, cfg, a.views, board, a.out)

    for eye, corners in views.items():
        if len(corners) < 6:
            print(f"[calib] {eye or 'mono'}: only {len(corners)} views, too few - skipped")
            continue
        size = sizes[eye]
        rms, K, D = fit(cv2, corners, size, board, a.square, cfg["model"])
        topic = cfg["topics"][0] if not eye else f"stereo/{eye}"
        out = a.out / (topic.replace("/", "_") + ".yaml")
        out.write_text(yaml.safe_dump({
            "model": cfg["model"],
            "image_width": int(size[0]),
            "image_height": int(size[1]),
            "K": K.tolist(),
            "D": D.tolist(),
            "rms_px": rms,
            "n_views": len(corners),
            "board": f"{bw}x{bh}",
            "square_m": a.square,
        }, sort_keys=False), encoding="utf-8")
        fx, fy = K[0, 0], K[1, 1]
        print(f"\n[calib] {topic}  rms={rms:.3f}px  from {len(corners)} views")
        print(f"        fx={fx:.1f} fy={fy:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print(f"        D={np.round(D, 5).tolist()}")
        print(f"        -> {out}")
        if rms > 1.0:
            print("        WARNING: rms above 1px - recapture with the board "
                  "closer to the frame edges and at stronger slants")
        if cfg["model"] == "fisheye" and "circle_radius" in cfg:
            # An equidistant fisheye maps its maximum field angle to the edge of
            # the image circle, so the calibrated f settles the lens's true FOV -
            # the 170-vs-180 question the MJCF has been guessing at.
            f_native = fx / cfg["downscale"]
            theta_max = cfg["circle_radius"] / f_native
            print(f"        implied lens FOV = {2*np.degrees(theta_max):.1f} deg "
                  f"(f_native={f_native:.1f} px/rad, circle r={cfg['circle_radius']:.0f} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
