"""Copy the CAD camera rig from one built scene to another.

Ports the mount geoms (head cap, both wrist brackets, the camera bodies) and
the five camera poses from a donor scene into a target scene, for both the
render and the sim file. Everything else in the target is left alone - this is
deliberately not a rebuild, because build_wbc_scene has previously come back
with the basket's mesh scales flattened.

Geoms are matched by their enclosing body name, which is the same in both
scenes apart from the `g1_` prefix the render file carries, so the rig lands on
the same links it came off.

    python scripts/realsim/port_cad_rig.py \
        --from outputs/room_a --to outputs/room_b
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

CAMERAS = ("head_pinhole", "elp_right", "elp_left",
           "ego_view_left_mono", "ego_view_right_mono")


def cad_geoms(lines, pre: str):
    """{body name without prefix: [geom lines]} for every CAD mesh geom."""
    out, body = {}, None
    for line in lines:
        m = re.search(r'<body name="([^"]+)"', line)
        if m:
            body = m.group(1)
            body = body[len(pre):] if pre and body.startswith(pre) else body
        if "<geom" in line and f'mesh="{pre}cad_' in line:
            out.setdefault(body, []).append(line)
    return out


def cad_meshes(lines, pre: str):
    return [l for l in lines if "<mesh" in l and f'name="{pre}cad_' in l]


def port(src: Path, dst: Path, pre_src: str, pre_dst: str) -> None:
    s, d = src.read_text().split("\n"), dst.read_text().split("\n")

    if any(f'mesh="{pre_dst}cad_' in l for l in d):
        raise SystemExit(f"{dst} already has the CAD rig")

    # --- mesh assets ------------------------------------------------------
    meshes = [l.replace(f'"{pre_src}cad_', f'"{pre_dst}cad_')
              for l in cad_meshes(s, pre_src)]
    for i, line in enumerate(d):
        if "</asset>" in line:
            d[i:i] = meshes
            break
    else:
        raise SystemExit("no </asset> in target")

    # --- geoms, keyed by the link they hang off ---------------------------
    want = cad_geoms(s, pre_src)
    placed, body = set(), None
    out = []
    for line in d:
        m = re.search(r'<body name="([^"]+)"', line)
        out.append(line)
        if m:
            raw = m.group(1)
            body = raw[len(pre_dst):] if pre_dst and raw.startswith(pre_dst) else raw
            if body in want and body not in placed:
                out.extend(l.replace(f'"{pre_src}cad_', f'"{pre_dst}cad_')
                           for l in want[body])
                placed.add(body)
    missing = set(want) - placed
    if missing:
        raise SystemExit(f"target has no body for: {', '.join(sorted(missing))}")
    d = out

    # --- camera poses -----------------------------------------------------
    moved = []
    for cam in CAMERAS:
        got = next((l for l in s if f'<camera name="{pre_src}{cam}"' in l), None)
        if got is None:
            raise SystemExit(f"donor has no camera {cam}")
        pos = re.search(r'pos="([^"]+)"', got).group(1)
        quat = re.search(r'quat="([^"]+)"', got).group(1)
        fovy = re.search(r'fovy="([^"]+)"', got).group(1)
        res = re.search(r'resolution="([^"]+)"', got)
        for i, line in enumerate(d):
            if f'<camera name="{pre_dst}{cam}"' not in line:
                continue
            new = re.sub(r'pos="[^"]+"', f'pos="{pos}"', line)
            new = re.sub(r'quat="[^"]+"', f'quat="{quat}"', new)
            new = re.sub(r'fovy="[^"]+"', f'fovy="{fovy}"', new)
            if res:
                new = re.sub(r'resolution="[^"]+"',
                             f'resolution="{res.group(1)}"', new)
            d[i] = new
            moved.append(cam)
            break
        else:
            raise SystemExit(f"target has no camera {cam}")

    dst.write_text("\n".join(d))
    print(f"  {dst.name}: {sum(len(v) for v in want.values())} geoms onto "
          f"{len(placed)} links, {len(moved)} cameras, {len(meshes)} meshes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--to", dest="dst", type=Path, required=True)
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for rel, pre in (("mujoco/scene_g1.xml", "g1_"),
                     ("mujoco/wbc/scene_wbc.xml", "")):
        s, t = a.src / rel, a.dst / rel
        shutil.copy2(t, t.with_suffix(f".xml.bak_cadrig_{stamp}"))
        port(s, t, pre, pre)
    print(f"backups .bak_cadrig_{stamp}")


if __name__ == "__main__":
    main()
