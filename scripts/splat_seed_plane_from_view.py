#!/usr/bin/env python3
"""Repair a planar surface in a splat from ONE inpainted view.

The rule every published method obeys, and every failed attempt here broke:
**supervise the hole from a single view, and get its geometry from somewhere
other than colour.** GScream, InFusion, SplatFill and RefFusion all inpaint one
reference view, lift it into 3D with a depth prior, and never let the other
views vote on what is inside the hole. SPIn-NeRF keeps all views but swaps the
pixel loss for patch-LPIPS, for the same reason from the other side.

What goes wrong otherwise is not bad luck, it is arithmetic. Give an L1 loss 450
mutually inconsistent targets for the same point and its minimiser is their
average; average 450 different inpaintings of a wood grain and you get brown
soup. That is failure mode 2, 3 and 4 of the earlier attempts, all at once.

Here the surface is a table, so the depth prior the papers work hard for is
free and exact: a plane. That removes the fragile part of those pipelines (
monocular depth completion, scale-and-shift alignment against a rendered depth
map) and leaves something that needs no optimisation at all:

    1. delete the gaussians the object owns - deleting, not masking, because a
       masked gaussian is never pruned either and simply sits there
    2. lay a grid over the hole, ON the plane, at the spacing you want
    3. project each grid point into the inpainted reference view and read its
       colour - exact, because a point on a known plane has a known projection
    4. drop a flat, disc-shaped gaussian at each point

Nothing is fitted, so nothing can be averaged, so nothing goes soft. The result
is exactly as sharp as the inpainter that produced the reference view, which is
why it is worth using a good one (LaMa) rather than OpenCV's.

    python scripts/splat_seed_plane_from_view.py \\
        --ply outputs/myroom/gs_bg.ply --out outputs/myroom/gs_bg_fixed.ply \\
        --scene-dir outputs/myroom --frame 00001.jpg \\
        --inpainted /path/lama_00001.png \\
        --centre 0.035 1.105 --radius 0.375 --height 0.75 \\
        --hole 0.165 1.033 0.13 --hole -0.006 1.061 0.18
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--frame", required=True,
                   help="Which colmap image is the reference view, e.g. 00001.jpg")
    p.add_argument("--inpainted", type=Path, required=True,
                   help="That frame with the objects painted out.")
    p.add_argument("--centre", type=float, nargs=2, required=True, metavar=("X", "Y"))
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--hole", type=float, nargs=3, action="append", default=[],
                   metavar=("X", "Y", "R"))
    p.add_argument("--spacing", type=float, default=0.0022,
                   help="Grid spacing on the surface, m. Near the reference "
                        "view's own footprint there - finer only interpolates.")
    p.add_argument("--thickness", type=float, default=0.02,
                   help="Flatness: in-plane extent over this is the disc's "
                        "aspect. Thin enough to be a surface, thick enough not "
                        "to alias.")
    p.add_argument("--opacity", type=float, default=0.97)
    p.add_argument("--no-flatten-lighting", action="store_true",
                   help="Keep the reference photo's own shading. One frame is "
                        "lit from one direction, so half the table is in shadow "
                        "and that shadow becomes part of the surface - fixed in "
                        "the scene while every other shadow moves. The splat it "
                        "replaces was averaged over a whole orbit and is evenly "
                        "lit, so by default the low-frequency brightness is "
                        "divided out and only the grain is kept.")
    p.add_argument("--repair-dark", type=float, default=0.62,
                   help="Replace samples darker than this fraction of the "
                        "surface's median. 0 disables.")
    p.add_argument("--light-scale", type=float, default=0.06,
                   help="Size, m, of the shading being removed. Well above the "
                        "grain, well below the table.")
    p.add_argument("--sink", type=float, default=0.003,
                   help="Drop the seeded sheet this far below the contact "
                        "plane, m. Objects rest at exactly `height`, so a sheet "
                        "AT that height is coplanar with their bottom faces and "
                        "the two fight for the same depth - which reads as a "
                        "haze creeping up the object. A few mm is invisible and "
                        "settles it.")
    p.add_argument("--no-match-brightness", action="store_true",
                   help="Skip matching the patch to the splat. Worth doing when "
                        "the patch replaces the WHOLE surface: there is then no "
                        "retained neighbour to seam against, and the reference "
                        "photograph's own colour is the more trustworthy of the "
                        "two - the splat's is an average over every view, "
                        "including each specular highlight the capture saw.")
    p.add_argument("--above", type=float, default=0.6,
                   help="Height of the slab of object to delete, m.")
    p.add_argument("--below", type=float, default=0.08,
                   help="How far below the surface to delete, m - the gaussians "
                        "that drew the tabletop under the object were fitted "
                        "through the object and are contaminated too.")
    a = p.parse_args()

    import cv2
    import pycolmap
    from plyfile import PlyData, PlyElement

    if not a.hole:
        raise SystemExit("give at least one --hole X Y R")
    scene = a.scene_dir

    # ---- where the reference camera is, in the scene's frame -------------
    rec = pycolmap.Reconstruction(str(scene / "roommesh" / "colmap" / "sparse" / "0"))
    img = next((i for i in rec.images.values() if i.name == a.frame), None)
    if img is None:
        raise SystemExit(f"{a.frame} is not in the reconstruction")
    cam = rec.cameras[img.camera_id]
    prm = cam.params
    K = np.array([[prm[0], 0, prm[2]], [0, prm[1], prm[3]], [0, 0, 1]], float)
    w2c = img.cam_from_world()
    R_c, t_c = w2c.rotation.matrix(), np.asarray(w2c.translation)

    tf = json.loads((scene / "splat_transform.json").read_text())
    Rt = np.asarray(tf["R"], float)
    s_ = float(tf["s"])
    t_ = np.asarray(tf["t"], float)

    def scene_to_pixel(P):
        """Scene metres -> reference-view pixels."""
        X = (P - t_) @ Rt / s_              # scene -> colmap  (R is orthonormal)
        cpt = X @ R_c.T + t_c
        z = cpt[:, 2]
        uv = np.stack([prm[0] * cpt[:, 0] / z + prm[2],
                       prm[1] * cpt[:, 1] / z + prm[3]], 1)
        return uv, z

    src = cv2.imread(str(a.inpainted))
    if src is None:
        raise SystemExit(f"cannot read {a.inpainted}")
    if (src.shape[1], src.shape[0]) != (cam.width, cam.height):
        src = cv2.resize(src, (cam.width, cam.height), interpolation=cv2.INTER_AREA)

    # ---- 1. delete what the object owns ----------------------------------
    ply = PlyData.read(str(a.ply))
    v = ply["vertex"]
    names = list(v.data.dtype.names)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    n0 = len(xyz)
    holes = np.asarray(a.hole, float).reshape(-1, 3)
    d_hole = np.stack([np.linalg.norm(xyz[:, :2] - h[:2], axis=1) - h[2]
                       for h in holes]).min(0)
    kill = (d_hole < 0) & (xyz[:, 2] > a.height - a.below) \
        & (xyz[:, 2] < a.height + a.above)
    print(f"[seed] deleting {kill.sum():,} gaussians ({100*kill.mean():.2f}%)")

    # ---- 2. a grid on the plane, inside the hole and on the surface ------
    k = int(np.ceil(max(h[2] for h in holes) / a.spacing)) + 2
    pts = []
    for h in holes:
        ii, jj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
        q = np.stack([h[0] + ii.ravel() * a.spacing,
                      h[1] + jj.ravel() * a.spacing], 1)
        q = q[np.linalg.norm(q - h[:2], axis=1) < h[2]]
        pts.append(q)
    grid = np.unique(np.round(np.vstack(pts) / a.spacing).astype(np.int64), axis=0) \
        * a.spacing
    grid = grid[np.linalg.norm(grid - np.asarray(a.centre), axis=1) < a.radius]
    P = np.column_stack([grid, np.full(len(grid), a.height - a.sink)])
    print(f"[seed] {len(P):,} points on the surface at {1000*a.spacing:.1f} mm")

    # ---- 3. read each point's colour out of the inpainted view -----------
    uv, z = scene_to_pixel(P)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width - 1) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height - 1)
    if ok.sum() < 100:
        raise SystemExit("the hole is barely visible in this reference view; "
                         "pick a frame that sees it")
    P, uv = P[ok], uv[ok]
    # bilinear, so the grid is not quantised to the source's pixel lattice
    # cv2.remap refuses maps longer than SHRT_MAX rows, so sample in chunks
    CH = 30000
    col = np.empty((len(uv), 3), np.float32)
    for i in range(0, len(uv), CH):
        q = uv[i:i + CH]
        col[i:i + CH] = cv2.remap(
            src, q[:, 0].astype(np.float32).reshape(-1, 1),
            q[:, 1].astype(np.float32).reshape(-1, 1),
            cv2.INTER_LINEAR).reshape(-1, 3)
    col = col[:, ::-1] / 255.0
    print(f"[seed] {len(P):,} points visible in {a.frame}")

    # Match the patch to the surface it abuts. The colours just read out carry
    # one frame's exposure, while the gaussians around them are an average over
    # many, and the step between the two draws a bright ring exactly along the
    # patch boundary - more conspicuous than whatever was being repaired.
    C0_ = 0.28209479177387814
    # Reference against the gaussians being REPLACED, not the ones beside them.
    # They are the splat's own view-averaged idea of this surface, there are
    # plenty of them, and matching their mean is what makes the new patch join
    # the old without a step along the boundary.
    ref = kill & (xyz[:, 2] > a.height - 0.04) & (xyz[:, 2] < a.height + 0.04)
    if a.no_match_brightness:
        print("[seed] brightness left as photographed")
    elif ref.sum() > 200:
        have = np.stack([v[f"f_dc_{i}"][ref] for i in range(3)], 1) * C0_ + 0.5
        # Median, not mean. On a splat trained with the objects still in it the
        # gaussians being replaced include the basket and the apple, and their
        # mean is far darker than the wood - matching it would dim the whole
        # table. They cover a fraction of the top, so the median ignores them.
        # Luminance only, one scalar. Matching per channel would also match the
        # splat's colour, and the splat's colour is an average that includes
        # every specular highlight the capture saw - paler and less saturated
        # than the wood really is. Brightness is the part that shows as a seam;
        # saturation is the part worth keeping from the photograph.
        w = np.array([0.114, 0.587, 0.299])          # col is RGB here
        gain = float(np.clip(np.median(have @ w[::-1]) / max(np.median(col @ w[::-1]), 1e-3),
                             0.6, 1.6))
        col = np.clip(col * gain, 0, 1)
        print(f"[seed] brightness-matched to the {ref.sum():,} gaussians it "
              f"replaces (gain {gain:.3f})")
    else:
        print(f"[seed] only {ref.sum()} reference gaussians - left unmatched")

    # Some samples land on things that are not the surface - a shadow cast
    # across it, a leg seen past its edge, whatever the inpainter left. They are
    # far darker than wood, so they are easy to name, and a round table gives an
    # exact replacement: the same radius, half a turn away.
    if a.repair_dark > 0:
        lum0 = col @ np.array([0.299, 0.587, 0.114])
        med = float(np.median(lum0))
        bad = lum0 < a.repair_dark * med
        if bad.any():
            rel2 = P[:, :2] - np.asarray(a.centre)
            opp = np.asarray(a.centre) - rel2            # 180 degrees round
            from scipy.spatial import cKDTree
            tree = cKDTree(P[~bad][:, :2])
            _, j = tree.query(opp[bad])
            col[bad] = col[~bad][j]
            print(f"[seed] repaired {int(bad.sum()):,} dark samples "
                  f"({100*bad.mean():.1f}%) from the opposite side")

    if not a.no_flatten_lighting:
        # Divide out the low-frequency illumination. The grid is a regular
        # lattice, so it rasterises exactly and a blur is a blur - no
        # resampling, and the high frequencies that carry the wood grain are
        # untouched.
        ij = np.round((P[:, :2] - P[:, :2].min(0)) / a.spacing).astype(np.int64)
        H_, W_ = ij[:, 1].max() + 1, ij[:, 0].max() + 1
        lum = np.zeros((H_, W_), np.float32)
        cnt = np.zeros((H_, W_), np.float32)
        w = np.array([0.299, 0.587, 0.114])
        np.add.at(lum, (ij[:, 1], ij[:, 0]), (col @ w).astype(np.float32))
        np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1.0)
        sig = max(1.0, a.light_scale / a.spacing)
        k = int(sig * 4) | 1
        num = cv2.GaussianBlur(lum, (k, k), sig)
        den = cv2.GaussianBlur(cnt, (k, k), sig)
        smooth = num / np.maximum(den, 1e-6)
        here = smooth[ij[:, 1], ij[:, 0]]
        target = float(np.median(here[here > 0]))
        adj = np.clip(target / np.maximum(here, 1e-4), 0.7, 1.45)
        col = np.clip(col * adj[:, None], 0, 1)
        print(f"[seed] flattened lighting over {a.light_scale*100:.0f} cm "
              f"(x{adj.min():.2f}..x{adj.max():.2f})")

    # ---- 4. one flat gaussian per point ----------------------------------
    m = len(P)
    keep = ~kill
    arr = np.empty(int(keep.sum()) + m, dtype=[(n, "f4") for n in names])
    for n in names:
        arr[n][:keep.sum()] = v[n][keep]
        arr[n][keep.sum():] = 0.0

    C0 = 0.28209479177387814
    arr["x"][keep.sum():] = P[:, 0].astype(np.float32)
    arr["y"][keep.sum():] = P[:, 1].astype(np.float32)
    arr["z"][keep.sum():] = P[:, 2].astype(np.float32)
    for i in range(3):
        arr[f"f_dc_{i}"][keep.sum():] = ((col[:, i] - 0.5) / C0).astype(np.float32)
    arr["opacity"][keep.sum():] = np.float32(
        np.log(a.opacity / (1.0 - a.opacity)))
    # a disc lying in the plane: two axes the size of the grid cell so
    # neighbours just touch, the third flattened so it reads as a surface
    lat = np.log(a.spacing * 0.72)
    arr["scale_0"][keep.sum():] = np.float32(lat)
    arr["scale_1"][keep.sum():] = np.float32(lat)
    arr["scale_2"][keep.sum():] = np.float32(np.log(a.spacing * a.thickness))
    # identity quaternion: the flattened axis is already z, and the surface is
    # horizontal, so no rotation is needed
    arr["rot_0"][keep.sum():] = np.float32(1.0)
    for i in (1, 2, 3):
        arr[f"rot_{i}"][keep.sum():] = np.float32(0.0)

    PlyData([PlyElement.describe(arr, "vertex")]).write(str(a.out))
    print(f"[seed] {n0:,} -> {len(arr):,} gaussians, wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
