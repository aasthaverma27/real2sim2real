#!/usr/bin/env python3
"""Evaluate several Gaussian splats against the same held-out frames.

Every splat is rendered at ONE fixed resolution against ONE ground-truth set,
because PSNR is not comparable across resolutions: a render at 720x1280 has
more high-frequency content to get wrong than the same scene at 540x960, so a
model trained and scored at higher resolution reports a lower number for
reasons that have nothing to do with quality. Comparing a 540x960 score against
a 720x1280 score - which is easy to do by accident when each training run
prints its own eval - measures the resolution, not the model.

Metrics:
  PSNR       standard, on the full frame
  SSIM       structural, less dominated by absolute brightness
  sharpness  high-frequency energy in the render / in the photo. 1.0 means the
             render is as sharp as reality; below 1.0 means it is blurrier.
             A model can gain PSNR while losing sharpness by blurring toward
             the mean, so PSNR alone cannot tell you whether detail improved.

The train/test split is regenerated with the same rule the trainer used (every
Nth frame held out), so the frames scored here were never trained on.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np


def load_splat(ply: Path, dev):
    import torch
    from plyfile import PlyData
    v = PlyData.read(str(ply))["vertex"]
    names = v.data.dtype.names
    col = lambda *k: np.column_stack([np.asarray(v[n], dtype=np.float32) for n in k])
    q = col("rot_0", "rot_1", "rot_2", "rot_3")
    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    sh0 = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    if n_rest:
        shN = np.stack([col(*[f"f_rest_{i}" for i in range(j, n_rest, 3)])
                        for j in range(3)], axis=2)
        sh = np.concatenate([sh0, shN], axis=1)
    else:
        sh = sh0
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    return {
        "means": t(col("x", "y", "z")), "quats": t(q),
        "scales": t(np.exp(col("scale_0", "scale_1", "scale_2"))),
        "opacities": t(1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))),
        "sh": t(sh), "deg": int(round(math.sqrt(sh.shape[1]) - 1)),
        "n": len(sh),
    }


def ssim(a, b):
    """Global SSIM on grayscale float images."""
    import cv2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2) /
         ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2)))
    return float(m.mean())


def main() -> int:
    import cv2, torch, pycolmap
    from gsplat import rasterization

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colmap", type=Path, required=True)
    p.add_argument("--ply", type=Path, nargs="+", required=True)
    p.add_argument("--label", nargs="+", default=None)
    p.add_argument("--eval-long-side", type=int, default=1280,
                   help="ONE resolution for every model. Native is 1280.")
    p.add_argument("--holdout", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all held-out frames")
    a = p.parse_args()

    labels = a.label or [q.parent.name for q in a.ply]
    if len(labels) != len(a.ply):
        raise SystemExit("--label count must match --ply count")

    rec = pycolmap.Reconstruction(str(a.colmap / "colmap" / "sparse" / "0"))
    cam = next(iter(rec.cameras.values()))
    # PINHOLE (4 params) is what colmap image_undistorter emits and needs no
    # distortion terms; OPENCV (8) is what feature extraction produces.
    if len(cam.params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    elif len(cam.params) == 4:
        fx, fy, cx, cy = cam.params
        k1 = k2 = p1 = p2 = 0.0
    elif len(cam.params) == 3:
        f_, cx, cy = cam.params
        fx = fy = f_
        k1 = k2 = p1 = p2 = 0.0
    else:
        raise SystemExit(f"unsupported camera model {cam.model.name}")
    Kf = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2])
    newK, _ = cv2.getOptimalNewCameraMatrix(Kf, dist, (cam.width, cam.height), 0,
                                            (cam.width, cam.height))
    m1, m2 = cv2.initUndistortRectifyMap(Kf, dist, None, newK,
                                         (cam.width, cam.height), cv2.CV_32FC1)
    sc = a.eval_long_side / max(cam.width, cam.height)
    W, H = int(round(cam.width * sc)), int(round(cam.height * sc))
    K = newK.copy(); K[0] *= W / cam.width; K[1] *= H / cam.height
    print(f"[eval] one fixed resolution for all models: {W}x{H}")

    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    test = [i for i in range(0, len(imgs), a.holdout)]
    if a.max_frames:
        test = test[:a.max_frames]
    print(f"[eval] {len(test)} held-out frames (every {a.holdout}th), "
          f"never seen in training")

    gts, views = [], []
    for i in test:
        img = imgs[i]
        raw = cv2.imread(str(a.colmap / "images" / img.name))
        if raw is None:
            continue
        u = cv2.resize(cv2.remap(raw, m1, m2, cv2.INTER_LINEAR), (W, H),
                       interpolation=cv2.INTER_AREA)
        gts.append(cv2.cvtColor(u, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        pose = img.cam_from_world()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = pose.rotation.matrix(); T[:3, 3] = pose.translation
        views.append(T)
    print(f"[eval] {len(gts)} frames loaded\n")

    dev = "cuda"
    Kt = torch.from_numpy(K.astype(np.float32)).to(dev)[None]
    rows = []
    for ply, lab in zip(a.ply, labels):
        if not ply.is_file():
            print(f"[eval] SKIP {lab}: {ply} missing"); continue
        g = load_splat(ply, dev)
        ps, ss, hi_r, hi_g = [], [], 0.0, 0.0
        for gt, T in zip(gts, views):
            with torch.no_grad():
                out, _, _ = rasterization(
                    means=g["means"], quats=g["quats"], scales=g["scales"],
                    opacities=g["opacities"], colors=g["sh"], sh_degree=g["deg"],
                    viewmats=torch.from_numpy(T).to(dev)[None], Ks=Kt,
                    width=W, height=H, packed=False)
            pr = out[0].clamp(0, 1).cpu().numpy()
            mse = float(np.mean((pr - gt) ** 2))
            ps.append(10 * math.log10(1.0 / max(mse, 1e-10)))
            pg = cv2.cvtColor(pr, cv2.COLOR_RGB2GRAY)
            gg = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
            ss.append(ssim(pg, gg))
            hi_r += float(np.abs(cv2.Laplacian(pg, cv2.CV_32F, ksize=3)).sum())
            hi_g += float(np.abs(cv2.Laplacian(gg, cv2.CV_32F, ksize=3)).sum())
            del out
        torch.cuda.empty_cache()
        rows.append((lab, g["n"], float(np.mean(ps)), float(np.std(ps)),
                     min(ps), float(np.mean(ss)), hi_r / hi_g,
                     ply.stat().st_size / 1e6))
        del g; torch.cuda.empty_cache()
        print(f"  {lab:24s} done")

    print(f"\n{'model':24s} {'gaussians':>12} {'PSNR':>8} {'sd':>6} {'worst':>7} "
          f"{'SSIM':>7} {'sharp':>7} {'MB':>7}")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: -x[2]):
        print(f"{r[0]:24s} {r[1]:>12,} {r[2]:>8.2f} {r[3]:>6.2f} {r[4]:>7.2f} "
              f"{r[5]:>7.4f} {r[6]:>7.3f} {r[7]:>7.0f}")
    print("\nsharp = render high-frequency energy / photo high-frequency energy; "
          "1.0 is as sharp as reality")
    return 0


if __name__ == "__main__":
    sys.exit(main())
