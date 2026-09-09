#!/usr/bin/env python3
"""Pick the sharpest, well-spread frames from a set of videos for COLMAP.

Sampling video at a fixed rate (`ffmpeg -vf fps=2`, as MuGS's guide suggests)
takes whatever frame lands on each tick, blur and all. With 18,922 frames
available and only a few hundred needed, that is a wasted choice: scoring every
candidate and keeping the sharpest costs one decode pass and removes the single
defect that COLMAP cannot recover from.

This matters more than usual for this footage. EXIF on the stills shows every
exposure between 1/25 s and 1/50 s against MuGS's >= 1/100 s guidance, so the
capture was light-starved and motion blur is the dominant defect - measurably
so, since the 12 MP stills score *below* the 2 MP video frames once both are
compared at the same resolution (674 vs 869).

**Temporal spread is enforced, not hoped for.** Sharpness alone would cluster
every pick on the few seconds where the camera paused, which is exactly the
degenerate near-still capture that produced a tilted, unusable splat earlier.
Each video is divided into equal time buckets and the sharpest frame from each
bucket wins, so coverage comes from the schedule and quality from the score.

Sharpness is measured at a fixed 960 px long edge. Laplacian variance scales
with resolution, so scoring sources of different sizes at their native size
compares the sizes rather than the footage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCORE_LONG_EDGE = 960


def sharpness(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    s = SCORE_LONG_EDGE / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def blown(img: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY)
    return float((g > 250).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True, help="images/ directory")
    p.add_argument("--total", type=int, default=400,
                   help="Frames to keep overall, shared out by video duration.")
    p.add_argument("--stride", type=int, default=3,
                   help="Score every Nth frame. 3 at 30 fps scores 10/s.")
    p.add_argument("--min-sharp", type=float, default=0.0,
                   help="Reject frames below this absolute score.")
    p.add_argument("--max-blown", type=float, default=1.0,
                   help="Reject frames with more than this fraction clipped.")
    p.add_argument("--report", type=Path, default=None)
    a = p.parse_args()

    vids = [Path(v) for v in a.videos]
    for v in vids:
        if not v.is_file():
            raise SystemExit(f"missing {v}")

    # share the budget by duration, so a 90 s sweep contributes more than a 12 s one
    durs = []
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        durs.append(n / fps)
    total_dur = sum(durs)
    budgets = [max(4, int(round(a.total * d / total_dur))) for d in durs]

    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("*.jpg"):
        old.unlink()

    kept, report = 0, []
    for v, budget in zip(vids, budgets):
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cand = []
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % a.stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    b = blown(frame)
                    if b <= a.max_blown:
                        s = sharpness(frame)
                        if s >= a.min_sharp:
                            cand.append((idx, s, b))
            idx += 1
        if not cand:
            print(f"[select] {v.name}: no usable frames", flush=True)
            cap.release()
            continue

        # one winner per time bucket: quality from the score, coverage from the schedule
        edges = np.linspace(0, n, budget + 1)
        picks = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inb = [c for c in cand if lo <= c[0] < hi]
            if inb:
                picks.append(max(inb, key=lambda c: c[1]))

        for rank, (fi, s, b) in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.retrieve() if False else cap.read()
            if not ok:
                continue
            name = f"{v.stem}_{fi:06d}.jpg"
            cv2.imwrite(str(a.out / name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            report.append({"video": v.name, "frame": int(fi),
                           "sharpness": round(s, 1), "blown": round(b, 4)})
            kept += 1
        cap.release()
        got = [c[1] for c in picks]
        allc = [c[1] for c in cand]
        print(f"[select] {v.name:<26} scored {len(cand):5d} -> kept {len(picks):3d}"
              f"  median sharp {np.median(got):6.0f} (pool median {np.median(allc):6.0f})",
              flush=True)

    print(f"\n[select] {kept} frames -> {a.out}")
    if report:
        s = np.array([r["sharpness"] for r in report])
        print(f"[select] kept-frame sharpness: median {np.median(s):.0f}, "
              f"p10 {np.percentile(s,10):.0f}, min {s.min():.0f}")
    if a.report:
        a.report.write_text(json.dumps(report, indent=2))
        print(f"[select] wrote {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
