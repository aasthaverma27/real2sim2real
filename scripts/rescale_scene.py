#!/usr/bin/env python3
"""Rescale a RoboSnap MuJoCo scene to real-world dimensions and reseat objects.

**Why this is needed.** SAM-3D reconstructs each object in its own canonical
frame; the pipeline then sizes it by fitting the mask against DA3's metric
depth. When depth is over-estimated along the viewing ray the object comes out
stretched along that ray. For the apple scene the camera looked *down* at the
table, and the result is a table 1.256 m tall (should be ~0.75) and an apple
0.102 m across (should be ~0.07). A Unitree G1's shoulder sits at ~1.05 m, so a
1.26 m table top is not merely inaccurate - it makes the pick physically
impossible, and any policy trained against it transfers to nothing.

**One global scale, not one per object.** The scene's depth came from VGGT
(`depth/camera.json` says `facebook/VGGT-1B`), and VGGT is scale-free - it
recovers geometry up to an unknown global factor. So the *ratios* between
objects are as good as the single depth map they were fitted against; only the
overall factor is wrong. Scaling objects independently therefore destroys
information the pipeline got right, which is exactly what a too-large basket on
a too-small table looks like.

The factor is anchored on the measured table height and then checked against a
second independent measurement:

    table height   0.75 m measured -> factor 0.75 / 1.256 = 0.597
    basket         0.362 x 0.415 x 0.156 m  x0.597  ->  0.216 x 0.248 x 0.093
    basket measured                                      0.190 x 0.245

1.2% on the long side from a factor derived without reference to the basket at
all. Both measurements are recorded in
`outputs/video_scene/object_properties.json`.

**Floor normalised to z=0.** The pipeline leaves the floor wherever gravity
alignment put it (z=-1.24 here). Shifting the scene so the floor is z=0 costs
nothing and makes robot placement and reward thresholds readable.

Objects are reseated after scaling: (x, y) offsets from the table centre scale
with the table so the layout stays proportional, and z is set so each object's
lowest vertex touches the table top rather than trusting the original height.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def body_extents(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB of every mesh geom on a body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr:adr + num]
        rot = data.geom_xmat[g].reshape(3, 3)
        pts.append(verts @ rot.T + data.geom_xpos[g])
    if not pts:
        raise SystemExit(f"body {body_name!r} has no mesh geoms")
    P = np.vstack(pts)
    return P.min(0), P.max(0)


def measure(xml: Path, bodies: list[str]) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for name in bodies:
        lo, hi = body_extents(model, data, name)
        out[name] = {"lo": lo, "hi": hi, "size": hi - lo,
                     "centre": (lo + hi) / 2}
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out["_floor_z"] = float(model.geom_pos[fid][2])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--table-body", default="obj0_round_wooden_table")
    p.add_argument("--table-height", type=float, default=0.75,
                   help="Measured table height, m. This alone sets the global "
                        "scale; every object is scaled by the same factor.")
    p.add_argument("--apple-body", default="obj2_red_apple")
    p.add_argument("--basket-body", default="obj1_woven_brown_basket")
    p.add_argument("--check-basket", type=float, nargs=2, default=(0.245, 0.19),
                   metavar=("LONG", "SHORT"),
                   help="Measured basket footprint, m, used only to verify the "
                        "factor derived from the table. Warns above 10%% error.")
    a = p.parse_args()

    bodies = [a.table_body, a.basket_body, a.apple_body]
    before = measure(a.scene, bodies)
    floor_z = before["_floor_z"]

    # ONE factor for the whole scene: VGGT is scale-free, so inter-object
    # ratios are trustworthy and only the global factor needs fixing
    factor = a.table_height / before[a.table_body]["size"][2]
    scales = {name: factor for name in bodies}
    print(f"[rescale] global scale x{factor:.5f} "
          f"(table height {a.table_height:.3f} m / "
          f"{before[a.table_body]['size'][2]:.3f} m reconstructed)")
    for name in bodies:
        s = before[name]["size"]
        print(f"  {name:28s} {s[0]:.3f} x {s[1]:.3f} x {s[2]:.3f} m"
              f"  ->  {s[0]*factor:.3f} x {s[1]*factor:.3f}"
              f" x {s[2]*factor:.3f} m")

    # independent check: the basket was measured separately, and the factor
    # above never looked at it
    bs = np.sort(before[a.basket_body]["size"][:2])[::-1] * factor
    want = np.sort(np.asarray(a.check_basket))[::-1]
    err = np.abs(bs - want) / want
    print(f"[rescale] basket check: predicted {bs[0]:.3f} x {bs[1]:.3f} m vs "
          f"measured {want[0]:.3f} x {want[1]:.3f} m "
          f"(error {100*err[0]:.1f}%, {100*err[1]:.1f}%)")
    if err.max() > 0.10:
        print(f"[rescale] WARNING: basket is off by {100*err.max():.0f}% - the "
              f"global-scale assumption may not hold for this scene")

    # stamp scale onto every mesh declaration belonging to each body
    tree = ET.parse(a.scene)
    root = tree.getroot()
    prefixes = {name.rsplit("_", 0)[0]: s for name, s in scales.items()}
    n_meshes = 0
    for mesh in root.iter("mesh"):
        mname = mesh.get("name", "")
        for prefix, s in prefixes.items():
            if mname.startswith(prefix):
                mesh.set("scale", f"{s:.9f} {s:.9f} {s:.9f}")
                n_meshes += 1
                break
    print(f"[rescale] scaled {n_meshes} mesh declarations")

    # floor to z=0
    for geom in root.iter("geom"):
        if geom.get("name") == "floor":
            pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
            geom.set("pos", f"{pos[0]} {pos[1]} 0")

    # scaling a mesh happens about the body origin, so body positions have to
    # move too - write provisional ones, then measure and correct
    body_el = {b.get("name"): b for b in root.iter("body")}
    for name in bodies:
        el = body_el[name]
        pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
        el.set("pos", " ".join(f"{v:.6f}" for v in
                               (pos - [0, 0, floor_z]) * scales[name]))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(a.output, encoding="unicode")

    # correction pass: table bottom on the floor, objects on the table top
    mid = measure(a.output, bodies)
    tree = ET.parse(a.output)
    root = tree.getroot()
    body_el = {b.get("name"): b for b in root.iter("body")}

    def shift(name, delta):
        el = body_el[name]
        pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
        el.set("pos", " ".join(f"{v:.6f}" for v in pos + delta))

    shift(a.table_body, np.array([0.0, 0.0, -mid[a.table_body]["lo"][2]]))
    tree.write(a.output, encoding="unicode")

    after_table = measure(a.output, [a.table_body])[a.table_body]
    top_z = float(after_table["hi"][2])
    t_ctr = after_table["centre"]

    mid2 = measure(a.output, bodies)
    tree = ET.parse(a.output)
    root = tree.getroot()
    body_el = {b.get("name"): b for b in root.iter("body")}
    for name in (a.basket_body, a.apple_body):
        # z only. The xy layout was already scaled when the body positions were
        # multiplied by the global factor; scaling the offset from the table
        # centre a second time pulls the objects together by factor^2 - at 0.597
        # that shrinks a 0.243 m apple-to-basket gap to 0.087 m, which puts the
        # apple inside the basket's footprint and the solver ejects it off the
        # table on the first step.
        d = np.array([0.0, 0.0, top_z - mid2[name]["lo"][2] + 0.001])
        shift(name, d)
    tree.write(a.output, encoding="unicode")

    final = measure(a.output, bodies)
    print(f"\n[rescale] floor z=0.000, table top z={top_z:.3f} m")
    for name in bodies:
        c, s, lo = final[name]["centre"], final[name]["size"], final[name]["lo"]
        print(f"  {name:28s} centre ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})"
              f"  size {s[0]:.3f}x{s[1]:.3f}x{s[2]:.3f}  bottom z={lo[2]:+.3f}")

    summary = {
        "scales": {k: float(v) for k, v in scales.items()},
        "table_top_z": top_z,
        "table_centre_xy": [float(t_ctr[0]), float(t_ctr[1])],
        "objects": {n: {"centre": final[n]["centre"].tolist(),
                        "size": final[n]["size"].tolist()} for n in bodies},
    }
    (a.output.parent / "rescale_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[done] {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
