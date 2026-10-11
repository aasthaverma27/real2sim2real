#!/usr/bin/env python3
"""Show the VLA camera streams, one window per camera.

vla_camera_server.py publishes exactly what the policy consumes - the ego view
and both wrist views - but nothing displayed them, so the wrist cameras could
only be checked by rendering stills offline. These are the frames that actually
reach the model, already resized to 640x480, so what you see here is what it
sees.

    vla_camera_viewer.py --port 5555 --topics hand_left,hand_right

Wire format is the server's own: [topic, header json, jpeg], three parts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np



# vla_camera_server's MSGPACK_KEYS, inverted: the msgpack protocol renames the
# wrist topics, so hand_left arrives as "ego_left" and hand_right as "ego_right".
MSGPACK_TO_TOPIC = {
    "ego_view": "stereo/right",
    "ego_left": "hand_left",
    "ego_right": "hand_right",
}


def _msgpack_images(frame):
    """Yield (name, jpeg_bytes) from a single-part msgpack camera frame."""
    import base64
    import msgpack
    try:
        msg = msgpack.unpackb(frame, raw=False)
    except Exception:
        return
    images = msg.get("images") if isinstance(msg, dict) else None
    if not isinstance(images, dict):
        return
    for name, val in images.items():
        if isinstance(val, str):
            try:
                val = base64.b64decode(val)
            except Exception:
                continue
        if isinstance(val, (bytes, bytearray)):
            yield name, bytes(val)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connect", default="tcp://127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--topics", default="hand_left,hand_right",
                   help="comma separated; the server publishes stereo/right, "
                        "hand_left, hand_right")
    p.add_argument("--pos", default="20,560;700,560",
                   help="window positions 'x,y;x,y', one per topic")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--topmost", action="store_true")
    p.add_argument("--timeout", type=float, default=30.0)
    a = p.parse_args()

    import cv2
    import zmq

    topics = [t.strip() for t in a.topics.split(",") if t.strip()]
    positions = []
    for chunk in a.pos.split(";"):
        try:
            x, y = chunk.split(",")
            positions.append((int(x), int(y)))
        except ValueError:
            positions.append(None)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"{a.connect}:{a.port}")
    for t in topics:
        sub.setsockopt(zmq.SUBSCRIBE, t.encode())
    # The server's 'msgpack' protocol sends ONE frame with no topic prefix, so a
    # topic-only subscription drops it. Subscribe to everything as well and sort
    # the two shapes out below.
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, int(a.timeout * 1000))
    print(f"[vla-view] {a.connect}:{a.port}  topics {topics}", flush=True)

    wins = {}
    for i, t in enumerate(topics):
        w = f"VLA {t}"
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        if i < len(positions) and positions[i]:
            cv2.moveWindow(w, *positions[i])
        if a.topmost:
            try:
                cv2.setWindowProperty(w, cv2.WND_PROP_TOPMOST, 1.0)
            except Exception:
                pass
        wins[t] = w

    seen = set()
    try:
        while True:
            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                print("[vla-view] no frames; is the server up with --vla?", flush=True)
                continue
            if len(parts) == 1:
                # 'msgpack' protocol: {"images": {ego_view|ego_left|ego_right:
                # base64 jpeg}}. Map those names back to the topic names.
                for name, payload in _msgpack_images(parts[0]):
                    topic = MSGPACK_TO_TOPIC.get(name)
                    if topic is None or topic not in wins:
                        continue
                    img = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if topic not in seen:
                        seen.add(topic)
                        print(f"[vla-view] {topic}: {img.shape[1]}x{img.shape[0]} "
                              f"(msgpack '{name}')", flush=True)
                    shown = cv2.resize(img, None, fx=a.scale, fy=a.scale) \
                        if a.scale != 1.0 else img
                    cv2.imshow(wins[topic], shown)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue
            if len(parts) != 3:
                continue
            topic = parts[0].decode()
            if topic not in wins:
                continue
            img = cv2.imdecode(np.frombuffer(parts[2], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if topic not in seen:
                seen.add(topic)
                h = json.loads(parts[1])
                print(f"[vla-view] {topic}: {img.shape[1]}x{img.shape[0]} "
                      f"camera={h.get('camera')}", flush=True)
            if a.scale != 1.0:
                img = cv2.resize(img, None, fx=a.scale, fy=a.scale)
            cv2.imshow(wins[topic], img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
