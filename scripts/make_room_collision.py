#!/usr/bin/env python3
"""Give the reconstructed room physical geometry, so the simulation has a room in it.

The splat is only ever visual: it has no presence in the physics, so a robot walks
through the real walls and objects fall past the real floor. Isaac Sim has the same
split - NuRec draws the room, USD geometry collides - and this is the MuJoCo half
of it: fit the room's dominant planes and emit them as MJCF geoms.

Planes rather than a convex decomposition of the 200k-face mesh, because a room IS
planes: one floor, a few walls, maybe a ceiling. Primitives are also far cheaper
for MuJoCo Warp than mesh-mesh collision.

    python scripts/make_room_collision.py --scene-dir outputs/video_scene \
        --out outputs/video_scene/mujoco/room_collision.xml

Include the result in a scene with:  <include file="room_collision.xml"/>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def load_points(scene_dir: Path, ply: Path | None, max_points: int,
                clip: float = 0.5):
    """Gaussian centres as a point cloud, trimmed to the dense core.

    A Lyra reconstruction carries far-field junk - this scene spans 138 m around a
    4 m room - so percentile clipping matters more than any fitting parameter.
    """
    from plyfile import PlyData

    if ply is None:
        cands = sorted((q for q in scene_dir.glob("gravity_aligned_background*.ply")
                        if "distilled" not in q.stem),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit(f"no background PLY in {scene_dir}")
        ply = cands[0]
    v = PlyData.read(str(ply))["vertex"]
    p = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    p = p[opacity > 0.05]
    lo, hi = np.percentile(p, [float(clip), 100.0 - float(clip)], axis=0)
    p = p[np.all((p >= lo) & (p <= hi), axis=1)]
    if len(p) > max_points:
        p = p[np.random.default_rng(0).choice(len(p), max_points, replace=False)]
    print(f"[room] {ply.name}: {len(p):,} points in the dense core, "
          f"extent {np.round(p.max(0) - p.min(0), 2).tolist()} m")
    return p, ply


def fit_planes(points, n_planes: int, tol: float, min_frac: float):
    """Sequential RANSAC: strongest plane, remove its inliers, repeat."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    found, remaining = [], pcd
    for _ in range(n_planes):
        if len(remaining.points) < max(200, min_frac * len(points)):
            break
        model, idx = remaining.segment_plane(distance_threshold=tol,
                                             ransac_n=3, num_iterations=2000)
        if len(idx) < min_frac * len(points):
            break
        inliers = np.asarray(remaining.points)[idx]
        found.append((np.asarray(model), inliers))
        remaining = remaining.select_by_index(idx, invert=True)
    return found


def merge_planes(planes, angle_deg: float, offset: float):
    """Fold together planes that describe the same surface.

    Sequential RANSAC keeps re-finding a big floor with slightly different
    parameters, which would emit three overlapping collision boxes for one floor.
    """
    cos_lim = np.cos(np.deg2rad(angle_deg))
    merged: list = []
    for model, inl in sorted(planes, key=lambda t: -len(t[1])):
        n = model[:3] / np.linalg.norm(model[:3])
        d = model[3] / np.linalg.norm(model[:3])
        for i, (m2, inl2) in enumerate(merged):
            n2 = m2[:3] / np.linalg.norm(m2[:3])
            d2 = m2[3] / np.linalg.norm(m2[:3])
            flip = 1.0 if np.dot(n, n2) >= 0 else -1.0
            if abs(np.dot(n, n2)) > cos_lim and abs(flip * d - d2) < offset:
                merged[i] = (m2, np.vstack([inl2, inl]))
                break
        else:
            merged.append((model, inl))
    if len(merged) != len(planes):
        print(f"[room] merged {len(planes)} raw planes into {len(merged)} surfaces")
    return merged


def classify(normal):
    """floor / ceiling / wall from the plane normal in a gravity-aligned frame."""
    nz = normal[2] / max(np.linalg.norm(normal), 1e-9)
    if nz > 0.85:
        return "floor"
    if nz < -0.85:
        return "ceiling"
    if abs(nz) < 0.35:
        return "wall"
    return "slanted"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--ply", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--planes", type=int, default=6, help="how many to look for")
    p.add_argument("--tol", type=float, default=0.04,
                   help="RANSAC inlier distance, metres")
    p.add_argument("--min-frac", type=float, default=0.02,
                   help="a plane must own at least this fraction of the points")
    p.add_argument("--thickness", type=float, default=0.05, help="wall box thickness")
    p.add_argument("--max-points", type=int, default=300_000)
    p.add_argument("--clip", type=float, default=0.5,
                   help="percentile clipped off each axis to drop far-field junk. "
                        "Too aggressive and the walls go with it.")
    p.add_argument("--best-only", action="store_true",
                   help="keep only the strongest surface per direction. RANSAC on a "
                        "reconstruction finds real surfaces and junk with equal "
                        "confidence; this drops everything but the dominant floor "
                        "and one wall per orientation.")
    p.add_argument("--max-span", type=float, default=6.0,
                   help="reject planes wider than this, metres - a plane larger "
                        "than the room is a bad fit, not a big wall")
    p.add_argument("--no-ceiling", action="store_true",
                   help="skip the ceiling, which usually just traps the camera")
    a = p.parse_args()

    pts, ply = load_points(a.scene_dir, a.ply, a.max_points, a.clip)
    planes = fit_planes(pts, a.planes, a.tol, a.min_frac)
    planes = merge_planes(planes, angle_deg=12.0, offset=0.12)
    if not planes:
        raise SystemExit("[room] no planes found - try a larger --tol")

    root = ET.Element("mujoco", model="room_collision")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="room")
    kept, seen_dirs = 0, []
    for model, inl in sorted(planes, key=lambda t: -len(t[1])):
        kind = classify(model[:3])
        if kind == "slanted" or (kind == "ceiling" and a.no_ceiling):
            print(f"  skip  {kind:8s} {len(inl):>7,} pts  (orientation)")
            continue
        if a.best_only:
            n = model[:3] / np.linalg.norm(model[:3])
            if any(abs(np.dot(n, s)) > np.cos(np.deg2rad(30.0)) for s in seen_dirs):
                print(f"  skip  {kind:8s} {len(inl):>7,} pts  (weaker duplicate "
                      f"of a surface already kept)")
                continue
            seen_dirs.append(n)
        n = model[:3] / np.linalg.norm(model[:3])
        if n[2] < 0:                       # point normals up/inward consistently
            n, model = -n, -model
        # The scene is gravity-aligned, so a floor is horizontal and a wall is
        # vertical - by definition, not by fit. RANSAC on a reconstruction returns
        # a floor tilted by a few degrees (11 degrees here), and a tilted floor is
        # a ramp: a ball lands on it, slides off the edge and falls forever.
        if kind in ("floor", "ceiling"):
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = np.array([n[0], n[1], 0.0])
            n /= max(np.linalg.norm(n), 1e-9)
        centre = inl.mean(axis=0)
        # In-plane extent, so a wall box covers the wall and no more. The basis
        # comes from the 3x3 covariance, not an SVD of the point matrix - a full
        # SVD of 150k points asks for a 174 GiB U.
        d = inl - centre
        # two in-plane axes orthogonal to the snapped normal
        seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = seed - np.dot(seed, n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        basis = np.vstack([u, v, n])
        span = np.abs(d @ basis[:2].T).max(axis=0)
        size = (float(max(span[0], 0.05)), float(max(span[1], 0.05)), a.thickness)
        if 2 * max(size[0], size[1]) > a.max_span:   # size is a half-extent
            print(f"  skip  {kind:8s} {len(inl):>7,} pts  "
                  f"({2 * max(size[:2]):.1f} m across - larger than the room, "
                  f"so a bad fit)")
            if a.best_only and seen_dirs:
                seen_dirs.pop()
            continue
        quat = plane_quat(n, basis[:2])
        centre = centre - n * a.thickness   # sink the slab so its face is the plane
        ET.SubElement(body, "geom", name=f"room_{kind}_{kept}", type="box",
                      pos=" ".join(f"{x:.4f}" for x in centre),
                      quat=" ".join(f"{x:.6f}" for x in quat),
                      size=" ".join(f"{x:.4f}" for x in size),
                      condim="3", friction="1.0 0.02 0.001",
                      rgba="0.55 0.55 0.6 0.35", group="3")
        print(f"  keep  {kind:8s} {len(inl):>7,} pts  centre "
              f"{np.round(centre, 2).tolist()}  size {np.round(size, 2).tolist()}")
        kept += 1

    if not kept:
        raise SystemExit("[room] nothing kept")
    out = a.out or a.scene_dir / "mujoco" / "room_collision.xml"
    ET.ElementTree(root).write(out)
    print(f"[room] wrote {out} with {kept} collision geoms")
    print(f"[room] add to a scene with:  <include file=\"{out.name}\"/>")
    return 0


def plane_quat(normal, in_plane):
    """MuJoCo (w,x,y,z) for a box whose local z is the plane normal."""
    z = normal / np.linalg.norm(normal)
    x = in_plane[0] - np.dot(in_plane[0], z) * z
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    tr = R.trace()
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                 (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                 (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                 (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    raise SystemExit(main())
