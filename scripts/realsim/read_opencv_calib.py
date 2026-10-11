#!/usr/bin/env python3
"""Read opencv_interactive-calibration's XML and say what it means for the sim.

Turns the tool's output into the numbers a MuJoCo camera needs, and grades it
against the checks that matter - not just the residual, which has already
proved it can look healthy on a calibration that is wrong.

    read_opencv_calib.py calib/stereo_right_opencv.xml --camera head
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# Every published stream leaves the camera server 16:9 with square pixels; the
# VLA client then squashes it to 640x480. That fixes fy/fx for any frame shape.
SOURCE_ASPECT = 16.0 / 9.0
VLA_OUT = (640, 480)
PUBLISHED = {"head": (768, 432), "wrist_left": (1280, 720),
             "wrist_right": (1280, 720)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xml", type=Path)
    p.add_argument("--camera", choices=sorted(PUBLISHED), default="head")
    p.add_argument("--write", type=Path, default=None,
                   help="also write the repo's calib yaml here")
    p.add_argument("--topic", default=None, help="topic name, for the yaml")
    p.add_argument("--fps", type=float, default=None,
                   help="measured publish rate, for the yaml")
    p.add_argument("--runs", type=float, nargs="*", default=None,
                   help="f_px from every run of this camera, so the yaml "
                        "records the real spread rather than the solver's "
                        "own standard deviation, which assumes independent "
                        "views and under-reports by roughly 20x here")
    a = p.parse_args()

    import cv2
    fs = cv2.FileStorage(str(a.xml), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"cannot read {a.xml}")
    K = fs.getNode("cameraMatrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    # cameraResolution is written as a 2-element SEQUENCE, not a matrix.
    rnode = fs.getNode("cameraResolution")
    if rnode.isSeq():
        res = np.array([rnode.at(0).real(), rnode.at(1).real()])
    else:
        res = np.asarray(rnode.mat()).ravel()
    rms = fs.getNode("avg_reprojection_error").real()
    nframes = int(fs.getNode("framesCount").real())
    Kstd = fs.getNode("cameraMatrix_std_dev").mat()
    Dstd = fs.getNode("dist_coeffs_std_dev").mat()
    date = fs.getNode("calibrationDate").string()
    fs.release()

    w, h = (int(res.ravel()[0]), int(res.ravel()[1]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    D = np.asarray(D).ravel()
    # std_dev rows are [fx, fy, cx, cy]; fx is 0 when the tool fixed the aspect
    ks = np.asarray(Kstd).ravel() if Kstd is not None else np.zeros(4)
    f_std = float(max(ks[0], ks[1])) if ks.size >= 2 else 0.0

    fovx = 2 * math.degrees(math.atan(0.5 * w / fx))
    fovy = 2 * math.degrees(math.atan(0.5 * h / fy))
    W, H = PUBLISHED[a.camera]
    ow, oh = VLA_OUT
    exp_aspect = SOURCE_ASPECT / (w / h)

    print(f"\n  {a.xml}   ({date.strip() or 'no date'})")
    print(f"  {'=' * 66}")
    print(f"  resolution   {w} x {h}      frames used {nframes}")
    print(f"  fx {fx:8.2f}   fy {fy:8.2f}   cx {cx:7.2f}   cy {cy:7.2f}")
    print(f"  D  {np.round(D, 5).tolist()}")
    if f_std:
        print(f"  focal std-dev  +-{f_std:.2f} px  ({100*f_std/fx:.1f}%)")

    checks = []

    def add(name, value, ok, want):
        checks.append((name, value, "PASS" if ok else "FAIL", want))

    add("reprojection error", f"{rms:.3f} px", rms < 0.5, "< 0.5 px")
    if f_std:
        add("focal uncertainty", f"+-{f_std:.2f} px ({100*f_std/fx:.1f}%)",
            f_std / fx < 0.02, "< 2%")
    got = fy / fx
    add(f"fy/fx vs predicted {exp_aspect:.4f}",
        f"{got:.4f} ({100*(got/exp_aspect-1):+.1f}%)",
        abs(got / exp_aspect - 1) < 0.03, "within 3%")
    dx, dy = cx - w / 2, cy - h / 2
    add("principal point offset", f"({dx:+.1f}, {dy:+.1f}) px",
        abs(dx) < 0.15 * w / 2 and abs(dy) < 0.15 * h / 2,
        f"< (+-{0.15*w/2:.0f}, +-{0.15*h/2:.0f})")
    dmax = float(np.max(np.abs(D)))
    add("distortion magnitude", f"max|D| = {dmax:.3f}", dmax < 1.0, "< 1.0")
    add("frames used", f"{nframes}", nframes >= 25, ">= 25")

    print()
    print(f"  {'CHECK':<34}{'VALUE':<26}VERDICT")
    print(f"  {'-' * 66}")
    for n, v, verdict, _ in checks:
        print(f"  {n:<34}{v:<26}{verdict}")

    print(f"\n  FOR THE SIM")
    print(f"    MuJoCo fovy     {fovy:.3f} deg     (fovx {fovx:.3f} deg)")
    print(f"    render at       {W} x {H}, then resize to {ow} x {oh}")
    if (w, h) != (W, H):
        print(f"    NOTE calibrated at {w}x{h}, published is {W}x{H}")
    print(f"    at {ow}x{oh}:   fx {fx*ow/w:.1f}  fy {fy*oh/h:.1f}  "
          f"cx {cx*ow/w:.1f}  cy {cy*oh/h:.1f}")

    if a.write is not None:
        import yaml
        doc = {
            "model": "pinhole",
            "topic": a.topic or a.write.stem,
            "image_width": w, "image_height": h,
            "K": [[float(K[i, j]) for j in range(3)] for i in range(3)],
            "D": [round(float(v), 8) for v in D],
            "rms_px": round(float(rms), 4),
            "n_views": nframes,
            "focal_std_px": round(f_std, 3),
            "source": "opencv_interactive-calibration on the published zmq "
                      "stream, via scripts/realsim/zmq_to_v4l2.py",
            "calibrated": date.strip().strip('"'),
            "board": "8x6 inner corners, taped flat to rigid card",
            "sim_camera": {
                "published_resolution": [W, H],
                "output_resolution": [ow, oh],
                "fovy_deg": round(fovy, 3),
                "fovx_deg": round(fovx, 3),
                "f_px_published": round(float(fx) * W / w, 2),
                "fx_out": round(float(fx) * ow / w, 2),
                "fy_out": round(float(fy) * oh / h, 2),
                "cx_out": round(float(cx) * ow / w, 2),
                "cy_out": round(float(cy) * oh / h, 2),
                "measured_fps": a.fps,
                "note": "MuJoCo has square pixels and one fovy, so it cannot "
                        "render output_resolution directly with both field "
                        "angles right. Render at published_resolution and "
                        "cv2.resize, the way the real VLA client does.",
            },
            "validation": {
                "checks": [{"check": n, "value": v, "verdict": vd}
                           for n, v, vd, _ in checks],
                "uncertainty_note":
                    "focal_std_px is calibrateCamera's linearised estimate. It "
                    "assumes the views are independent; a hand-held board gives "
                    "correlated poses, so it under-reports. Trust "
                    "runs_spread_f_px below, measured by repeating the "
                    "calibration.",
            },
        }
        if a.runs:
            fovs = [2 * math.degrees(math.atan(0.5 * H / f)) for f in a.runs]
            doc["validation"]["runs_f_px"] = [round(f, 2) for f in a.runs]
            doc["validation"]["runs_spread_f_px"] = [round(min(a.runs), 2),
                                                     round(max(a.runs), 2)]
            doc["validation"]["runs_fovy_band_deg"] = [round(min(fovs), 2),
                                                       round(max(fovs), 2)]
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\n  wrote {a.write}")

    fails = [c for c in checks if c[2] == "FAIL"]
    if fails:
        print(f"\n  {len(fails)} CHECK(S) FAILED:")
        for n, v, _, want in fails:
            print(f"    - {n}: {v}   (want {want})")
    else:
        print("\n  ALL CHECKS PASSED")
    return 0 if not fails else 2


# $RS_PY_SIM (configs/paths.env) has cv2, yaml and zmq.
_FALLBACK_PYTHONS = tuple(p for p in (__import__("os").environ.get("RS_PY_SIM"),
                                      "/usr/bin/python3") if p)


def ensure_deps() -> None:
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass
    import os
    me = Path(sys.executable).resolve()
    for cand in _FALLBACK_PYTHONS:
        if not Path(cand).exists() or Path(cand).resolve() == me:
            continue
        if subprocess.run([cand, "-c", "import cv2"],
                          capture_output=True).returncode == 0:
            os.execv(cand, [cand, str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit("need opencv-python")


if __name__ == "__main__":
    ensure_deps()
    sys.exit(main())
