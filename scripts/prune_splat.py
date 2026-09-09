#!/usr/bin/env python3
"""Delete the Gaussians occupied by simulated bodies from a background splat.

Anything that exists as a MuJoCo body must not also exist in the background,
or MuGS composites two copies of it: the mesh one and the photographed one.
The depth test cannot separate them because they sit at the same depth.

For each named body this takes the convex hull of its visual mesh in world
coordinates, pushes every hull plane outward by a margin (the splat's copy is
never perfectly co-located with the mesh), and drops the Gaussians whose centre
falls inside. A convex hull is the right shape here even for a table: the hull
of a disc on four legs is the whole volume the table occupies, which is exactly
what has to go.

The floor under a body was never observed during capture - the body occluded it
- so pruning leaves a hole there. --keep-below preserves the ground band so the
floor itself survives, and the mesh body covers what remains from any sane
viewpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def body_hull(model, data, bid, mujoco):
    """World-space vertices of every visual mesh geom on a body subtree."""
    pts = []
    stack = [bid]
    while stack:
        b = stack.pop()
        for c in range(model.nbody):
            if int(model.body_parentid[c]) == b and c != b:
                stack.append(c)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != b:
                continue
            if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mi = int(model.geom_dataid[g])
            adr, n = int(model.mesh_vertadr[mi]), int(model.mesh_vertnum[mi])
            v = model.mesh_vert[adr:adr + n]
            pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    if not pts:
        return None
    return np.concatenate(pts, 0).astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--body", action="append", required=True,
                   metavar="NAME[:MARGIN]",
                   help="body to remove from the splat; margin in metres "
                        "(default 0.05)")
    p.add_argument("--keep-below", type=float, default=0.025,
                   help="never prune Gaussians below this height, so the floor "
                        "survives under the bodies (default 0.025 m)")
    a = p.parse_args()

    import mujoco
    from plyfile import PlyData, PlyElement
    from scipy.spatial import ConvexHull

    model = mujoco.MjModel.from_xml_path(str(a.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ply = PlyData.read(str(a.ply))
    vert = ply["vertex"]
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    n0 = len(xyz)
    print(f"[prune] {n0:,} Gaussians in {a.ply.name}")

    drop = np.zeros(n0, bool)
    for spec in a.body:
        name, _, m = spec.partition(":")
        margin = float(m) if m else 0.05
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"no body named {name!r}")
        pts = body_hull(model, data, bid, mujoco)
        if pts is None:
            raise SystemExit(f"body {name!r} has no visual mesh")
        hull = ConvexHull(pts)
        # equations are [n | d] with n.x + d <= 0 inside; +margin dilates
        A, d = hull.equations[:, :3], hull.equations[:, 3]

        # A hull of a detailed mesh has thousands of facets, and testing ten
        # million Gaussians against all of them is minutes of arithmetic. The
        # bounding box throws away >99% of them first, so only the handful that
        # could possibly be inside ever reach the plane test.
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        cand = np.all((xyz >= lo) & (xyz <= hi), axis=1) & (xyz[:, 2] > a.keep_below)
        idx = np.flatnonzero(cand)
        sub = xyz[idx]
        ok = np.ones(len(idx), bool)
        for i in range(0, len(A), 256):
            ok &= np.all(sub @ A[i:i + 256].T + d[i:i + 256] <= margin, axis=1)
        inside = np.zeros(n0, bool)
        inside[idx[ok]] = True
        print(f"[prune] {name:32s} margin {margin:.3f} m  hull {len(A)} planes"
              f"  bbox-candidates {len(idx):,}"
              f"  -> {inside.sum():,} Gaussians ({100 * inside.mean():.2f}%)")
        drop |= inside

    keep = ~drop
    print(f"[prune] dropping {drop.sum():,} of {n0:,} "
          f"({100 * drop.mean():.2f}%), keeping {keep.sum():,}")

    out = PlyElement.describe(vert.data[keep], "vertex")
    PlyData([out], text=False).write(str(a.out))
    print(f"[prune] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
