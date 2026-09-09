#!/usr/bin/env python3
"""Rigidly move a splat so it lines up with a different scene's coordinates.

RoboSnap splats are gravity-aligned to the room they were captured in. Anything
that expects the splat to sit in *its* world - MuGS's mjlab tasks, for instance,
where a table-mounted arm works just above z=0 - needs the splat transformed, and
neither MuGS's sensor config nor mjlab has a field for it. So the transform is
baked into a copy of the PLY.

A 3D gaussian is exactly rigid-transformable: rotate the mean, quaternion-multiply
the rotation, leave the scale. Our PLYs are DC-only so there are no spherical
harmonic bands to rotate.

    # put the reconstructed table top at the origin, room extending away from +x
    python scripts/place_splat.py --ply <in.ply> --out <out.ply> \
        --origin 0.094,0.697,0.0137 --yaw-deg 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_mul(a, b):
    """Hamilton product of (w,x,y,z) rows."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--origin", required=True, metavar="X,Y,Z",
                   help="point in the splat that becomes the target origin")
    p.add_argument("--yaw-deg", type=float, default=0.0,
                   help="rotation about z applied after recentring")
    p.add_argument("--lift", type=float, default=0.0,
                   help="extra z offset applied last, metres")
    a = p.parse_args()

    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(a.ply))
    v = ply["vertex"].data.copy()
    c = np.array([float(x) for x in a.origin.split(",")], dtype=np.float64)
    th = np.deg2rad(a.yaw_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])

    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    xyz = (xyz - c) @ R.T
    xyz[:, 2] += a.lift
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if all(f"rot_{i}" in v.dtype.names for i in range(4)):
        q = np.column_stack([v[f"rot_{i}"] for i in range(4)]).astype(np.float64)
        q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        qr = np.tile(np.array([[np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)]]), (len(q), 1))
        q = quat_mul(qr, q)
        for i in range(4):
            v[f"rot_{i}"] = q[:, i]

    if any(n.startswith("f_rest_") for n in v.dtype.names) and abs(a.yaw_deg) > 1e-9:
        print("[place] WARNING: this PLY carries spherical harmonics, which are NOT "
              "rotated here - view-dependent shading will be wrong. Distil to "
              "degree 0 first, or extend this script with a Wigner-D rotation.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(a.out))
    print(f"[place] {len(v):,} gaussians: origin {c.tolist()} -> (0,0,{a.lift}), "
          f"yaw {a.yaw_deg:g} deg")
    print(f"[place] new bounds x[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] "
          f"y[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] "
          f"z[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")
    print(f"[place] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
