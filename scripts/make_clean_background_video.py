#!/usr/bin/env python3
"""Remove the foreground objects from a capture video, for background reconstruction.

RoboSnap's design keeps the background object-free: preprocess masks the objects
out and inpaints the holes, and the orbit is generated from that clean image. If
the background is instead reconstructed from raw footage, the objects end up baked
into the splat and appear twice - once as simulated geometry, once as background.

This applies the same idea to real footage: union the per-frame GUI masks, dilate,
inpaint, and write an object-free video to reconstruct from.

    python scripts/make_clean_background_video.py \
        --video 20260826_095303.mp4 --masks outputs/robot_video/multi_mask \
        --output inputs/background_clean.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True, help="multi_mask directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--objects", default="", help="comma list; default = every object dir")
    p.add_argument("--dilate", type=int, default=9, help="pixels, to cover mask edges/shadow")
    p.add_argument("--radius", type=int, default=6, help="inpaint radius")
    p.add_argument("--scale", type=float, default=0.5,
                   help="inpaint at this scale for speed, then blend at full res")
    a = p.parse_args()

    objs = ([o.strip() for o in a.objects.split(",") if o.strip()]
            or sorted(d.name for d in a.masks.iterdir()
                      if (d / "all_mask").is_dir()))
    print(f"[clean] removing {len(objs)} objects: {', '.join(objs)}")

    mask_files = {o: sorted((a.masks / o / "all_mask").glob("*.png")) for o in objs}
    n = min(len(v) for v in mask_files.values())
    print(f"[clean] {n} masked frames available")

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # cv2 ignores a video's rotation metadata, while the GUI applies it - so the
    # masks can be portrait while the decoded frames are landscape. Rotate the
    # FRAME to the masks' orientation; resizing across it would misalign badly.
    probe = cv2.imread(str(next(iter(mask_files.values()))[0]), cv2.IMREAD_UNCHANGED)
    mh, mw = probe.shape[:2]
    rot = None
    if first.shape[:2] != (mh, mw):
        for name, code in (("cw", cv2.ROTATE_90_CLOCKWISE),
                           ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
                           ("180", cv2.ROTATE_180)):
            if cv2.rotate(first, code).shape[:2] == (mh, mw):
                rot = code
                print(f"[clean] rotating frames {name} to match masks "
                      f"({first.shape[1]}x{first.shape[0]} -> {mw}x{mh})")
                break
        if rot is None:
            raise SystemExit(f"frame {first.shape[:2]} cannot be rotated to mask {(mh, mw)}")
    H, W = (mh, mw)
    print(f"[clean] output {W}x{H} @ {fps:.1f} fps")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    kern = np.ones((a.dilate, a.dilate), np.uint8)
    sw, sh = int(W * a.scale), int(H * a.scale)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= n:
            break
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        m = np.zeros((H, W), np.uint8)
        for o in objs:
            mi = cv2.imread(str(mask_files[o][i]), cv2.IMREAD_UNCHANGED)
            if mi is None:
                continue
            if mi.shape[:2] != (H, W):
                raise SystemExit(f"mask {mi.shape[:2]} != frame {(H, W)} for {o}")
            alpha = mi[:, :, 3] if mi.ndim == 3 and mi.shape[2] == 4 else mi[:, :, 0]
            m |= (alpha > 127).astype(np.uint8) * 255
        m = cv2.dilate(m, kern, iterations=1)

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, msmall, a.radius, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (W, H), interpolation=cv2.INTER_LINEAR)

        out = frame.copy()
        out[m > 0] = filled[m > 0]
        writer.write(out)
        if i % 100 == 0:
            print(f"[clean] frame {i}/{n}  masked {100*(m>0).mean():.1f}%")
        i += 1

    cap.release(); writer.release()
    print(f"[clean] wrote {a.output} ({i} frames, {a.output.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
