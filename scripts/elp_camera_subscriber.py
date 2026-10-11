#!/usr/bin/env python3
"""Subscribe to the ELP fisheye streams - the client side, for checking a feed.

Two separate windows, one per lens, because the publisher sends two topics and
a real stereo consumer treats them as two cameras. Also reports whether the
pair is actually synchronous, which is the one property of this module that is
easy to break and easy to miss.

    scripts/elp_camera_subscriber.py --port 5560            # two windows
    scripts/elp_camera_subscriber.py --port 5560 --save /tmp/elp  # frames to disk
    scripts/elp_camera_subscriber.py --port 5560 --no-window --frames 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", default="tcp://127.0.0.1")
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--side", choices=["left", "right", "both"], default="both")
    p.add_argument("--frames", type=int, default=0, help="0 = until ESC/Ctrl-C")
    p.add_argument("--save", type=Path, help="write frames here instead of/as well")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--pos-left", type=int, nargs=2, default=[30, 60])
    p.add_argument("--pos-right", type=int, nargs=2, default=[700, 60])
    p.add_argument("--topmost", action="store_true")
    a = p.parse_args()

    import cv2
    import zmq

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"{a.connect}:{a.port}")
    sides = ["left", "right"] if a.side == "both" else [a.side]
    proj = "rectilinear"      # overwritten by the stream header
    for s in sides:
        sub.setsockopt(zmq.SUBSCRIBE, f"elp/{s}".encode())
    sub.setsockopt(zmq.RCVTIMEO, int(a.timeout * 1000))
    print(f"[sub] connected to {a.connect}:{a.port}, topics {sides}")

    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
    wins = {}
    if not a.no_window:
        for s in sides:
            # the publisher states its projection in the stream header,
            # so the window says what it is actually showing
            w = f"ELP {proj} - {s} lens"
            cv2.namedWindow(w, cv2.WINDOW_NORMAL)
            pos = a.pos_left if s == "left" else a.pos_right
            cv2.moveWindow(w, pos[0], pos[1])
            if a.topmost:
                try:
                    cv2.setWindowProperty(w, cv2.WND_PROP_TOPMOST, 1.0)
                except Exception:
                    pass
            wins[s] = w

    stamps, n, first = {}, 0, True
    try:
        while a.frames == 0 or n < a.frames:
            try:
                topic, hdr, payload = sub.recv_multipart()
            except zmq.Again:
                print(f"[sub] nothing for {a.timeout}s - is the publisher up?")
                break
            h = json.loads(hdr)
            side = h["side"]
            if h["encoding"] == "jpeg":
                img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(payload, np.uint8).reshape(h["height"], h["width"], 3)
            if first:
                print(f"[sub] stream info: {json.dumps({k: v for k, v in h.items() if k not in ('side','frame_id','stamp_ns','nbytes')})}")
                proj = h.get("projection", proj)
                first = False
            stamps[side] = (h["frame_id"], h["stamp_ns"])
            if len(stamps) == 2:
                (fl, sl), (fr, sr) = stamps["left"], stamps["right"]
                if fl == fr:
                    print(f"[sub] frame {fl}: pair synchronous, "
                          f"stamp delta {abs(sl - sr) / 1e6:.3f} ms, "
                          f"{h['nbytes'] / 1024:.0f} kB/eye")
                    stamps.clear()
            if a.save:
                cv2.imwrite(str(a.save / f"{side}_{h['frame_id']:06d}.png"), img)
            if side in wins:
                cv2.imshow(wins[side], img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        if wins:
            cv2.destroyAllWindows()
        sub.close(0); ctx.term()
    print(f"[sub] received {n} messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
