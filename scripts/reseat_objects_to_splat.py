#!/usr/bin/env python3
"""Move the mesh objects onto the splat, using frame 0 as the referee.

The object reconstruction and the splat disagree about where things are, and the
splat is the one to believe. VGGT recovers depth up to a factor that varies
along the viewing ray (see `rescale_scene.py`), so the objects come out the
right size - their size was anchored on a tape measurement - at the wrong
distance. In our worked example the table landed 0.18 m nearer the camera than it
really is. COLMAP has no such freedom: it triangulated those positions from 450
views that had to agree with each other.

The fix needs no new information, because frame 0 is in both reconstructions.
Each object's mask says exactly which direction it lies in from that camera, and
the splat says how far away the surface it rests on is. Direction plus surface
is a position:

    table        ->  the splat's own tabletop, found as the dense horizontal
                     disc at the height the tape says it should be
    on the table ->  ray through the mask centroid, met with that tabletop plane

Heights are not touched: the floor is z=0 and the table top is the measured
height above it, both already true before this runs.

    python scripts/reseat_objects_to_splat.py --scene-dir outputs/myroom \\
        --table-body obj0_round_wooden_table --table-height 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--table-body", required=True)
    p.add_argument("--table-height", type=float, required=True)
    p.add_argument("--scene-xml", type=Path, default=None,
                   help="default <scene-dir>/mujoco/scene_scaled.xml, edited in place")
    p.add_argument("--ply", type=Path, default=None,
                   help="default <scene-dir>/gs_bg.ply - the PLACED splat")
    p.add_argument("--band", type=float, default=0.13,
                   help="Half-thickness, m, of the slice taken around the "
                        "tabletop height when looking for it in the splat.")
    p.add_argument("--min-opacity", type=float, default=0.2)
    p.add_argument("--max-move", type=float, default=0.6,
                   help="Refuse to move a body further than this. A large jump "
                        "means the mask and the splat are describing different "
                        "things, not that the object was very badly placed.")
    a = p.parse_args()

    import mujoco
    from PIL import Image
    from plyfile import PlyData

    scene = a.scene_dir
    xml = a.scene_xml or (scene / "mujoco" / "scene_scaled.xml")
    tf = json.loads((scene / "splat_transform.json").read_text())
    C = np.asarray(tf["camera_in_scene"], float)
    # The camera's pose and the lens, both as the ALIGNER settled them: its
    # rotation carries the floor-levelling correction, and its intrinsics are
    # COLMAP's rather than VGGT's. Reading gravity_alignment.json here instead
    # would silently reintroduce both errors the aligner exists to remove.
    R_gc = np.asarray(tf["camera_rotation_in_scene"], float)
    fx, fy, ux, uy = tf["intrinsics_at_mask_res"]

    def ray(u, v):
        d = R_gc @ np.array([(u - ux) / fx, (v - uy) / fy, 1.0])
        return d / np.linalg.norm(d)

    def on_plane(u, v, z):
        """Where this pixel's ray crosses the horizontal plane at height z."""
        d = ray(u, v)
        if abs(d[2]) < 1e-6:
            return None
        return C + d * ((z - C[2]) / d[2])

    # ---- the splat's own tabletop ---------------------------------------
    ply = PlyData.read(str(a.ply or (scene / "gs_bg.ply")))["vertex"]
    P = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    opa = 1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64)))
    P = P[opa > a.min_opacity]
    top = a.table_height
    # A thin slice through the surface holds almost nothing: a tabletop is a
    # few gaussians deep, and the RIM - which is what gets fitted below - hangs
    # below the surface by the thickness of the top. Take the whole top edge.
    band = P[(P[:, 2] > top - a.band) & (P[:, 2] < top + a.band)]
    if len(band) < 400:
        raise SystemExit(f"only {len(band)} gaussians near z={top:.2f} - the splat "
                         f"has no tabletop there; is --table-height right?")
    # Find the tabletop by fitting its RIM, not by averaging the surface.
    # Averaging fails in a way that looks like it worked: the apple and the
    # basket were masked out of the splat, so the surface has a hole in it
    # exactly where they stood, and the mean of what is left is pulled off
    # centre - by 0.26 m in this scene, which is a third of the table.
    #
    # The rim has no such problem. It is a circle, every point on it is real,
    # and a vote over candidate centres finds it without any initial guess: the
    # right centre is the one that makes the most points share a radius.
    def _vote(pts, cxs, cys, rlo, rhi, step):
        best = None
        for cx in cxs:
            for cy in cys:
                r = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                r = r[(r > rlo) & (r < rhi)]
                if len(r) < 100:
                    continue
                h, e = np.histogram(r, bins=int((rhi - rlo) / step),
                                    range=(rlo, rhi))
                k = int(h.argmax())
                sc = int(h[max(0, k - 1):k + 2].sum())
                if best is None or sc > best[0]:
                    best = (sc, cx, cy, float(e[k]) + step / 2)
        return best

    def fit_rim(pts, c0, span=1.3, rlo=0.20, rhi=0.60):
        """Centre and radius of a round tabletop, from the rim alone.

        Coarse to fine, and deliberately not seeded: the densest patch of a
        tabletop is somewhere on it, not in the middle of it, so anything that
        starts from a seed and refines locally converges to the wrong circle.
        A vote over a wide grid has no such preference - the true centre is
        simply the one that puts the most points at a common radius.
        """
        c = _vote(pts, np.arange(c0[0] - span / 2, c0[0] + span / 2, 0.02),
                  np.arange(c0[1] - span / 2, c0[1] + span / 2, 0.02),
                  rlo, rhi, 0.01)
        if c is None:
            return None
        f = _vote(pts, np.arange(c[1] - 0.03, c[1] + 0.03, 0.005),
                  np.arange(c[2] - 0.03, c[2] + 0.03, 0.005),
                  rlo, rhi, 0.005)
        _, cx, cy, rad = f or c
        r = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        on = np.abs(r - rad) < 0.02
        ang = np.degrees(np.arctan2(pts[on, 1] - cy, pts[on, 0] - cx))
        cover = len(np.unique((ang // 10).astype(int))) / 36.0
        return np.array([cx, cy]), rad, cover, int(on.sum())

    cell = np.round(band[:, :2] / 0.08).astype(int)
    (gx, gy), _ = Counter(map(tuple, cell)).most_common(1)[0]
    seed = np.array([gx * 0.08, gy * 0.08])
    local = band[np.linalg.norm(band[:, :2] - seed, axis=1) < 1.2][:, :2]
    print(f"[reseat] {len(local):,} points in the tabletop slice near "
          f"({seed[0]:+.2f},{seed[1]:+.2f})")
    fit = fit_rim(local, seed)
    if fit is not None and fit[2] >= 0.6:
        t_xy, t_r, cover, n_on = fit
        print(f"[reseat] tabletop rim: {n_on:,} points, {100*cover:.0f}% of the "
              f"circle, radius {t_r:.3f} m")
    else:
        # not a round table, or too little rim to be sure of - fall back to the
        # midpoint of its extents, which at least ignores interior holes
        near = band[np.linalg.norm(band[:, :2] - seed, axis=1) < 0.6]
        lo2 = np.percentile(near[:, :2], 1, axis=0)
        hi2 = np.percentile(near[:, :2], 99, axis=0)
        t_xy = (lo2 + hi2) / 2
        t_r = float(np.mean(hi2 - lo2)) / 2
        got = f"{100*fit[2]:.0f}%" if fit else "no fit"
        print(f"[reseat] no usable rim ({got}); using the extents instead")

    print(f"[reseat] splat tabletop at ({t_xy[0]:+.3f}, {t_xy[1]:+.3f}), "
          f"{2*t_r:.3f} m across")

    # ---- masks, in the order the bodies were named ------------------------
    masks = sorted((scene / "sam3d").glob("*.png"), key=lambda q: int(q.stem))
    names = [ln.strip() for ln in
             (scene / "object.txt").read_text().splitlines() if ln.strip()]

    model = mujoco.MjModel.from_xml_path(str(xml))
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                  for b in range(model.nbody)]

    # each body's own height, so an object can be placed by its centre rather
    # than by the front edge a bottom-of-mask ray actually lands on
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def body_box(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            pts.append(model.mesh_vert[adr:adr + num]
                       @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        if not pts:
            return None
        Q = np.vstack(pts)
        return Q.min(0), Q.max(0)

    tree = ET.parse(xml)
    el = {b.get("name"): b for b in tree.getroot().iter("body")}
    moves = {}
    for i, mpath in enumerate(masks):
        body = next((n for n in body_names if n and n.startswith(f"obj{i}_")), None)
        if body is None or body not in el:
            print(f"[reseat] mask {mpath.name} has no obj{i}_ body; skipped")
            continue
        old = np.array([float(x) for x in el[body].get("pos", "0 0 0").split()])
        # a body's origin is not its middle - the table's is 1.5 cm off - so
        # aim the MESH at the target and let the origin fall where it must
        bb = body_box(body)
        skew = ((bb[0][:2] + bb[1][:2]) / 2 - old[:2]) if bb is not None else np.zeros(2)
        if body == a.table_body:
            new_xy = t_xy
        else:
            alpha = np.array(Image.open(mpath).convert("RGBA"))[..., 3] > 128
            ys, xs = np.nonzero(alpha)
            if len(xs) < 50:
                print(f"[reseat] {body}: mask is nearly empty; skipped")
                continue
            box = body_box(body)
            if box is None:
                continue
            half = float(box[1][2] - box[0][2]) / 2.0
            # the ray through the middle of the mask, met with the plane through
            # the middle of the object. Meeting it with the tabletop instead
            # would return the point where the object's front face touches down,
            # which for the basket is 10 cm from its centre.
            hit = on_plane(xs.mean(), ys.mean(), top + half)
            if hit is None:
                continue
            new_xy = hit[:2]
        d = float(np.linalg.norm(new_xy - (old[:2] + skew)))
        if d > a.max_move:
            print(f"[reseat] {body}: refusing to move {d:.3f} m "
                  f"(limit {a.max_move}) - left where it was")
            continue
        origin = new_xy - skew
        moves[body] = (old[:2] + skew, new_xy, d)
        el[body].set("pos", f"{origin[0]:.6f} {origin[1]:.6f} {old[2]:.6f}")

    if not moves:
        print("[reseat] nothing moved")
        return 0
    for b, (o, n, d) in moves.items():
        print(f"  {b:28s} ({o[0]:+.3f},{o[1]:+.3f}) -> "
              f"({n[0]:+.3f},{n[1]:+.3f})   moved {d:.3f} m")
    tree.write(xml, encoding="unicode")
    print(f"[reseat] wrote {xml}")

    # the table is the one body whose size can be checked against the splat,
    # and a mismatch here is the reconstruction's proportions, not its placement
    box = body_box(a.table_body)
    if box is not None:
        lo, hi = box
        w = float(np.mean(hi[:2] - lo[:2]))
        if True:
            err = (w - 2 * t_r) / (2 * t_r)
            print(f"[reseat] table mesh is {w:.3f} m across, the splat's is "
                  f"{2*t_r:.3f} m ({100*err:+.0f}%)")
            if abs(err) > 0.12:
                print("[reseat] the two disagree by more than a tenth. Size was "
                      "anchored on the measured HEIGHT alone, so a round table "
                      "reconstructed with the wrong aspect keeps the error in "
                      "its diameter. Measure the diameter too and rebuild.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
