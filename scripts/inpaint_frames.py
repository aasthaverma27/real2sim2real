#!/usr/bin/env python3
"""Paint masked objects out of training frames, so the splat never sees a hole.

Dropping masked pixels from the training loss is the obvious way to keep an
object out of a background, and for an object that MOVES it is the right one:
the patch it hides in one view is visible in the next, so the optimiser still
learns what is behind it.

A basket sitting on a table does not move. It hides the same patch of tabletop
in every frame of the capture, and masking that patch out of the loss means no
view ever constrains it. Gaussians initialised there get no gradient, so they
are neither corrected nor pruned, and what survives to the end is whatever the
initialisation happened to put there - which renders as a smear sitting on the
table, worse than the ghost object the mask was there to prevent.

Inpainting removes the hole instead of creating one. Every frame gets plausible
tabletop where the object was, the views disagree slightly, and 3DGS averages
that disagreement into a smooth surface - which on a uniform wooden table is
indistinguishable from the real thing.

**Copy the surface from other frames, do not invent it.** Telea invents a smooth
fill, and it invents a slightly different one in every frame. 3DGS reconciles
that disagreement the only way it can - by averaging - and the result is a soft
blurred patch lying on the table, which is what the object used to be. The way
out is that the table is *planar* and the camera *orbits* it: every tabletop
point hidden behind the basket in this frame is in plain view in a frame forty
away. Two views of a plane are related by a homography, exactly, so that real
texture can be warped in. The poses give that homography to within a few pixels
- close, but a few pixels of misregistration is itself a blur - so each one is
then refined against the actual pixels of the table the two frames share, and
kept only if that refinement converges. `--colmap` plus `--plane-from` turns this on; the fill
is then real pixels, consistent between views, and the splat reconstructs it as
sharply as the rest of the table.

**Fill each surface from itself.** An object standing on a table hides part of
the table AND part of whatever is behind it, so its mask straddles the table's
far rim. Inpainting that in one go drags floor grey down onto the tabletop in a
wedge, which is worse than the object was. Given `--surface-masks` - a per-frame
mask of the table - the hole is split at the rim and each side is filled only
from pixels on its own side, by marking everything on the other side unknown so
the fill cannot reach it.

    python scripts/inpaint_frames.py \\
        --images outputs/myroom/roommesh/images \\
        --masks  outputs/myroom/roommesh/masks \\
        --out    outputs/myroom/roommesh/images_clean

The poses do not change, so the existing COLMAP reconstruction is reused as is;
only `train_gs.py --image-dir` moves to the new directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--surface-masks", type=Path, default=None,
                   help="Per-frame mask of the surface the objects stand on. "
                        "Without it the fill crosses the surface's edge and "
                        "smears the background across it.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dilate", type=int, default=6,
                   help="Extra pixels beyond the mask. The masks are dilated "
                        "already; this covers the object's contact shadow, "
                        "which is part of the object as far as the splat is "
                        "concerned and stays behind if it is left in.")
    p.add_argument("--colmap", type=Path, default=None,
                   help="<scene>/roommesh - poses for the planar fill.")
    p.add_argument("--plane-from", type=Path, default=None,
                   help="splat_transform.json; with --surface-height it gives "
                        "the surface's plane in colmap coordinates.")
    p.add_argument("--surface-height", type=float, default=0.75,
                   help="Height of that surface in the built scene, m.")
    p.add_argument("--donors", type=int, default=8,
                   help="How many other frames to try per hole. They are taken "
                        "from around the orbit, so each sees the table from a "
                        "clearly different angle.")
    p.add_argument("--radius", type=int, default=8, help="cv2.inpaint radius")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Inpaint at this scale and upsample the result into the "
                        "hole. Inpainting is slow and the fill is smooth, so "
                        "full resolution costs minutes and buys nothing.")
    a = p.parse_args()

    import cv2

    frames = sorted(list(a.images.glob("*.jpg")) + list(a.images.glob("*.png")))
    if not frames:
        raise SystemExit(f"no images in {a.images}")
    a.out.mkdir(parents=True, exist_ok=True)
    kern = (np.ones((a.dilate * 2 + 1,) * 2, np.uint8) if a.dilate > 0 else None)

    # ---- planar multi-view fill ------------------------------------------
    poses = None
    if a.colmap is not None and a.plane_from is not None:
        import json
        import pycolmap
        rec = pycolmap.Reconstruction(str(a.colmap / "colmap" / "sparse" / "0"))
        cam = next(iter(rec.cameras.values()))
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        Ki = np.linalg.inv(K)
        tfj = json.loads(a.plane_from.read_text())
        Rt = np.asarray(tfj["R"], float)
        sc = float(tfj["s"])
        tt = np.asarray(tfj["t"], float)
        # scene z = s*(R X)_z + t_z, so the scene's horizontal plane at
        # `surface-height` is the colmap plane n.X = c with these:
        n = Rt[2, :].copy()
        c = (a.surface_height - tt[2]) / sc
        poses = {}
        for im in rec.images.values():
            w2c = im.cam_from_world()
            poses[im.name] = (w2c.rotation.matrix(), np.asarray(w2c.translation))
        order = sorted(poses)
        idx = {nm: i for i, nm in enumerate(order)}
        print(f"[inpaint] planar fill on, {len(poses)} poses, "
              f"plane n={np.round(n,3)} c={c:.3f}")

        def refine(H, tgt_img, don_img, tab):
            """Nudge the pose-derived homography onto the pixels.

            The poses are good but not exact, and a two-pixel error copied into
            the hole is a blur with extra steps. ECC maximises correlation over
            the table the two frames can both see, at quarter resolution
            because that is where the table's large-scale texture lives and it
            keeps this affordable. If it will not converge the pair is dropped
            rather than used at whatever it had reached.
            """
            f_ = 0.25
            g1 = cv2.cvtColor(cv2.resize(tgt_img, None, fx=f_, fy=f_), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(cv2.resize(don_img, None, fx=f_, fy=f_), cv2.COLOR_BGR2GRAY)
            msk = cv2.resize(tab.astype(np.uint8), (g1.shape[1], g1.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            if msk.sum() < 400:
                return H
            S_ = np.diag([f_, f_, 1.0])
            Hs = (S_ @ H @ np.linalg.inv(S_)).astype(np.float32)
            Hs /= Hs[2, 2]
            try:
                _, Hs = cv2.findTransformECC(
                    g1, g2, Hs, cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                    msk, 5)
            except cv2.error:
                return None
            return np.linalg.inv(S_) @ Hs.astype(np.float64) @ S_

        def homography(src, dst):
            """Pixels in `dst`'s image -> pixels in `src`'s image, via the plane."""
            Rd, td = poses[dst]
            Rs, ts = poses[src]
            Rrel = Rs @ Rd.T
            trel = ts - Rrel @ td
            m = (n @ Rd.T) @ Ki                  # row vector
            aa = c + n @ (Rd.T @ td)
            return aa * (K @ Rrel @ Ki) + np.outer(K @ trel, m)

    done = filled = 0
    planar = leftover = 0
    areas = []
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        if img is None:
            continue
        mp = a.masks / (f.stem + ".png")
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.is_file() else None
        if m is None or not m.any():
            # nothing to remove in this frame - copy it through unchanged rather
            # than round-tripping it via an encoder for no reason
            cv2.imwrite(str(a.out / f.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            done += 1
            continue
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        m = (m > 127).astype(np.uint8)
        if kern is not None:
            m = cv2.dilate(m, kern)
        areas.append(float(m.mean()))

        sm = None
        if a.surface_masks is not None:
            sp = a.surface_masks / (f.stem + ".png")
            sm = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
            if sm is not None:
                if sm.shape[:2] != img.shape[:2]:
                    sm = cv2.resize(sm, (img.shape[1], img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                sm = (sm > 127).astype(np.uint8)
                # The segmenter excludes the objects FROM the table, so the
                # table comes back with object-shaped holes in it - and then
                # every pixel to be filled counts as "not table" and gets
                # filled from the floor. Close the enclosed holes: the patch
                # under a basket is table. A notch in the silhouette's edge,
                # where the basket stands proud of the far rim, is not enclosed
                # and correctly stays outside.
                cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                sm = np.zeros_like(sm)
                cv2.drawContours(sm, cnts, -1, 1, thickness=cv2.FILLED)

        def fill(hole, forbid=None, margin=140):
            """Inpaint `hole`, allowed to copy from anywhere except `forbid`.

            Cropped to the hole's neighbourhood first. Marking every pixel off
            the table "unknown" is what stops the fill reaching across the rim,
            but done over the whole frame it asks Telea to invent most of the
            image; it only ever needed the band of real table around the hole.
            """
            ys_, xs_ = np.nonzero(hole)
            if len(xs_) == 0:
                return None, None
            x0 = max(0, xs_.min() - margin); x1 = min(img.shape[1], xs_.max() + margin)
            y0 = max(0, ys_.min() - margin); y1 = min(img.shape[0], ys_.max() + margin)
            sub = img[y0:y1, x0:x1]
            unk = hole[y0:y1, x0:x1]
            if forbid is not None:
                unk = np.maximum(unk, forbid[y0:y1, x0:x1])
            small = cv2.resize(sub, None, fx=a.scale, fy=a.scale,
                               interpolation=cv2.INTER_AREA)
            us = cv2.resize(unk, (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            if not us.any() or us.all():
                return None, None
            r = cv2.inpaint(small, us, a.radius, cv2.INPAINT_TELEA)
            r = cv2.resize(r, (sub.shape[1], sub.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            return r, (y0, y1, x0, x1)

        fixed = img.copy()

        def apply(hole, forbid):
            r, box = fill(hole, forbid)
            if r is None:
                return
            y0, y1, x0, x1 = box
            h = hole[y0:y1, x0:x1][..., None] > 0
            fixed[y0:y1, x0:x1] = np.where(h, r, fixed[y0:y1, x0:x1])

        if sm is None or not sm.any():
            apply(m, None)
        else:
            on = (m & sm).astype(np.uint8)
            off = (m & (1 - sm)).astype(np.uint8)

            if poses is not None and f.name in poses and on.any():
                # Real texture first: warp other views of the same plane in.
                need = on.copy()
                me = idx[f.name]
                N = len(order)
                # spread the donors around the orbit rather than taking
                # neighbours, which see almost exactly what this frame sees and
                # therefore have the same pixels hidden
                # Moderate baselines, both ways round. Far enough that the
                # basket's shadow has swung off the patch it hid here, near
                # enough that the table is still seen face-on rather than edge-
                # on, which is where resampling starts costing sharpness.
                steps = []
                for k in range(1, a.donors // 2 + 1):
                    d0 = int(round(N * 0.06 * k))
                    steps += [d0, -d0]
                tabsrc = ((sm > 0) & (m == 0))
                for st in steps:
                    if not need.any():
                        break
                    dn = order[(me + st) % N]
                    dimg = cv2.imread(str(a.images / dn))
                    if dimg is None:
                        continue
                    H = homography(dn, f.name)
                    H = refine(H, img, dimg, tabsrc)
                    if H is None:
                        continue
                    # H maps THIS frame's pixels to the donor's, which is the
                    # inverse of what warpPerspective assumes by default
                    warp = cv2.warpPerspective(
                        dimg, H, (img.shape[1], img.shape[0]),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                    # the donor's own object mask, warped the same way: a pixel
                    # hidden there too is no use
                    dm = cv2.imread(str(a.masks / (Path(dn).stem + ".png")),
                                    cv2.IMREAD_GRAYSCALE)
                    dsm = cv2.imread(str(a.surface_masks / (Path(dn).stem + ".png")),
                                     cv2.IMREAD_GRAYSCALE) if a.surface_masks else None
                    ok = np.ones(dimg.shape[:2], np.uint8)
                    if dm is not None:
                        ok &= (dm <= 127).astype(np.uint8)
                    if dsm is not None:
                        ok &= (dsm > 127).astype(np.uint8)
                    okw = cv2.warpPerspective(
                        ok, H, (img.shape[1], img.shape[0]),
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                    use = (need > 0) & (okw > 0)
                    if use.sum() < 20:
                        continue
                    # auto-exposure drifts between frames, so match the donor to
                    # this frame over the table they share before copying
                    both = tabsrc & (okw > 0)
                    if both.sum() > 500:
                        g = (img[both].astype(np.float32).mean(0) /
                             np.maximum(warp[both].astype(np.float32).mean(0), 1e-3))
                        warp = np.clip(warp.astype(np.float32) * g, 0, 255).astype(np.uint8)
                    fixed[use] = warp[use]
                    need[use] = 0
                    planar += int(use.sum())
                if need.any():
                    apply(need, (1 - sm).astype(np.uint8))
                    leftover += int(need.sum())
            elif on.any():
                apply(on, (1 - sm).astype(np.uint8))
            if off.any():
                apply(off, sm)
        # feather the seam: a hard edge between real and invented pixels is a
        # high-frequency line, and a splat will happily reconstruct it
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
        out = (img * (1 - alpha) + fixed * alpha).astype(np.uint8)
        cv2.imwrite(str(a.out / f.name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1
        filled += 1
        if done % 50 == 0 or done == len(frames):
            print(f"  {done}/{len(frames)}", flush=True)

    print(f"\n[inpaint] {done} frames -> {a.out}")
    print(f"[inpaint] {filled} had something to remove, "
          f"mean {100*np.mean(areas) if areas else 0:.2f}% of the frame")
    if planar or leftover:
        tot = planar + leftover
        print(f"[inpaint] {100*planar/max(tot,1):.1f}% of the surface was filled "
              f"with real pixels from other views, {100*leftover/max(tot,1):.1f}% "
              f"had to be invented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
