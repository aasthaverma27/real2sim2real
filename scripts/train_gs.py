#!/usr/bin/env python3
"""Train a 3D Gaussian Splat from a COLMAP reconstruction - the Re3Sim route.

Re3Sim's `reconstruct.py` hands its COLMAP output to the INRIA 3DGS trainer
(`diff-gaussian-rasterization` + `simple-knn`), and MuGS's own data-collection
guide prescribes the same thing: COLMAP SfM, then 30k iterations of 3DGS
training. Neither of those CUDA extensions is built in this workspace, but
gsplat 1.5.2 is - and `gsplat.strategy.DefaultStrategy` *is* the 3DGS paper's
densify/split/prune/opacity-reset schedule, with the paper's own defaults
(prune_opa 0.005, grow_grad2d 0.0002, refine 500-15000 every 100, reset every
3000). So this is the reference recipe, not an approximation of it.

**Why this and not the feed-forward path.** RoboSnap's background comes from
Lyra: VIPE poses plus Depth-Anything-3 depth, fused in a single forward pass
with no optimiser anywhere. Its gaussian count is exactly
`da3_max_frames x 642,656` - going from 240 to 674 input frames changed it by
zero. An orbit therefore *smears*, because per-view depths that disagree get
averaged rather than reconciled. Optimisation is the missing step.

A previous attempt optimised only colour and opacity on Lyra's fixed geometry.
It gained 4.36 dB of PSNR and looked worse: with no densification or pruning,
the cheapest way to reduce L1 is to inflate gaussians and blur toward the mean.
Densification is not a refinement of that idea, it is the part that makes it
work.

**Distortion is not skipped.** The fitted OPENCV model has k1=0.0103,
k2=-0.0049 - about 8 px at the image corner. Small, but larger than the 0.47 px
mean reprojection error the reconstruction achieved, so images are undistorted
and pinhole intrinsics used thereafter.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_colmap(sparse_dir: Path, image_dir: Path, long_side: int,
                mask_dir: Path | None = None):
    """Return (images uint8 NHWC, w2c Nx4x4, K 3x3, points xyz, points rgb, names,
    masks uint8 NHW or None).

    A mask is white where an object should be ignored. It goes through the same
    undistortion and resize as its image - a mask that is merely resized is off
    by the distortion, which at the frame corner is about 8 px.
    """
    import cv2
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    cam = next(iter(rec.cameras.values()))
    # OPENCV cameras carry 8 params; a PINHOLE camera - what
    # colmap image_undistorter emits - carries 4 and needs no distortion terms,
    # because the images it wrote are already undistorted.
    if len(cam.params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    elif len(cam.params) == 4:
        fx, fy, cx, cy = cam.params
        k1 = k2 = p1 = p2 = 0.0
    elif len(cam.params) == 3:                      # SIMPLE_PINHOLE / SIMPLE_RADIAL
        f_, cx, cy = cam.params
        fx = fy = f_
        k1 = k2 = p1 = p2 = 0.0
    else:
        raise SystemExit(f"unsupported camera model {cam.model.name} "
                         f"with {len(cam.params)} params")
    print(f"[gs] camera {cam.model.name}: fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f} dist={'none' if not any((k1,k2,p1,p2)) else 'yes'}")
    K_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2], dtype=np.float64)

    # one undistortion map for every frame - a single camera was enforced at
    # extraction time, so the map is shared
    newK, _ = cv2.getOptimalNewCameraMatrix(
        K_full, dist, (cam.width, cam.height), 0, (cam.width, cam.height))
    map1, map2 = cv2.initUndistortRectifyMap(
        K_full, dist, None, newK, (cam.width, cam.height), cv2.CV_32FC1)

    scale = long_side / max(cam.width, cam.height)
    W, H = int(round(cam.width * scale)), int(round(cam.height * scale))
    K = newK.copy()
    K[0] *= W / cam.width
    K[1] *= H / cam.height

    imgs, w2cs, names, masks = [], [], [], ([] if mask_dir else None)
    missing_masks = 0
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        path = image_dir / img.name
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        und = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        imgs.append(cv2.cvtColor(cv2.resize(und, (W, H), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB))
        if masks is not None:
            mp = mask_dir / (Path(img.name).stem + ".png")
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                # a frame with no mask is a frame with no object in it, which is
                # a legitimate answer - supervise all of it
                missing_masks += 1
                m = np.zeros((cam.height, cam.width), np.uint8)
            m = cv2.remap(m, map1, map2, cv2.INTER_NEAREST)
            masks.append((cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127))
        pose = img.cam_from_world()          # pycolmap 4.x: a method, not a field
        T = np.eye(4)
        T[:3, :3] = pose.rotation.matrix()
        T[:3, 3] = pose.translation
        w2cs.append(T)
        names.append(img.name)

    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0

    # COLMAP always leaves a few far-flung points; they would set a scene extent
    # orders of magnitude too large and cripple the position learning rate
    ctr = np.median(xyz, axis=0)
    keep = np.linalg.norm(xyz - ctr, axis=1) < np.percentile(
        np.linalg.norm(xyz - ctr, axis=1), 99.0)
    print(f"[colmap] trimmed {(~keep).sum():,} outlier points "
          f"({100 * (~keep).mean():.1f}%)")

    mstack = None
    if masks is not None:
        mstack = np.stack(masks)
        print(f"[masks] {mstack.mean() * 100:.2f}% of pixels excluded from the loss"
              f"{f', {missing_masks} frame(s) had no mask' if missing_masks else ''}")

    return (np.stack(imgs), np.stack(w2cs).astype(np.float32), K.astype(np.float32),
            xyz[keep], rgb[keep], names, mstack)


def knn_scale(xyz: np.ndarray, k: int = 4) -> np.ndarray:
    """Initial gaussian size = mean distance to k nearest neighbours."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    return np.clip(d[:, 1:].mean(axis=1), 1e-7, None)


def gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """SSIM on CHW tensors. fused_ssim is not installed, so this is the plain form."""
    C = a.shape[0]
    w = win.expand(C, 1, *win.shape[-2:])
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a[None], w, padding=pad, groups=C)
    mu_b = F.conv2d(b[None], w, padding=pad, groups=C)
    aa = F.conv2d(a[None] * a[None], w, padding=pad, groups=C) - mu_a ** 2
    bb = F.conv2d(b[None] * b[None], w, padding=pad, groups=C) - mu_b ** 2
    ab = F.conv2d(a[None] * b[None], w, padding=pad, groups=C) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (aa + bb + C2)
    return (num / den).mean()


def save_ply(path: Path, params: dict) -> None:
    from plyfile import PlyData, PlyElement
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    opac = params["opacities"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(means), -1)
    shN = params["shN"].detach().cpu().numpy().reshape(len(means), -1)

    cols = [("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    cols += [(f"f_dc_{i}", "f4") for i in range(sh0.shape[1])]
    cols += [(f"f_rest_{i}", "f4") for i in range(shN.shape[1])]
    cols += [("opacity", "f4")]
    cols += [(f"scale_{i}", "f4") for i in range(3)]
    cols += [(f"rot_{i}", "f4") for i in range(4)]
    arr = np.empty(len(means), dtype=cols)
    arr["x"], arr["y"], arr["z"] = means.T
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(sh0.shape[1]):
        arr[f"f_dc_{i}"] = sh0[:, i]
    for i in range(shN.shape[1]):
        arr[f"f_rest_{i}"] = shN[:, i]
    arr["opacity"] = opac
    for i in range(3):
        arr[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def main() -> int:
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colmap", type=Path, required=True,
                   help="Work dir from colmap_from_video.py (holds images/ and colmap/).")
    p.add_argument("--out", type=Path, required=True, help="Output .ply")
    p.add_argument("--iters", type=int, default=30000,
                   help="30000 is the reference schedule (the iteration_30000 "
                        "checkpoints MuGS ships are exactly this).")
    p.add_argument("--long-side", type=int, default=960,
                   help="Training resolution on the LONG image edge. The source "
                        "video is 720x1280, so 1280 is native; anything less "
                        "throws away pixels.")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="Override <colmap>/images.")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--holdout", type=int, default=8,
                   help="Every Nth frame is held out, the INRIA convention.")
    p.add_argument("--masks", type=Path, default=None,
                   help="Directory of <frame stem>.png binary masks, white where "
                        "an object is. Those pixels are dropped from the loss, so "
                        "the splat is built only from views where the object is "
                        "not in the way - no object baked in, and nothing to prune "
                        "out afterwards. scripts/mask_frames_sam3.py writes them.")
    p.add_argument("--vram-limit-gb", type=float, default=31.0,
                   help="Abort before exceeding this, rather than OOM mid-run.")
    a = p.parse_args()

    dev = "cuda"
    torch.manual_seed(0)

    imgs, w2c, K, xyz, rgb, names, masks = load_colmap(
        a.colmap / "colmap" / "sparse" / "0",
        a.image_dir or (a.colmap / "images"), a.long_side, a.masks)
    N, H, W = imgs.shape[:3]
    print(f"[data] {N} frames at {W}x{H}, fx={K[0,0]:.1f}, "
          f"{len(xyz):,} init points")

    test = set(range(0, N, a.holdout))
    train_idx = np.array([i for i in range(N) if i not in test])
    test_idx = np.array(sorted(test))
    print(f"[data] {len(train_idx)} train / {len(test_idx)} held out")

    cams = np.stack([-T[:3, :3].T @ T[:3, 3] for T in w2c])
    scene_scale = float(np.linalg.norm(cams - cams.mean(0), axis=1).max()) * 1.1
    print(f"[data] scene scale (camera spread) {scene_scale:.3f}")

    # ---- init ----
    s = np.log(knn_scale(xyz))[:, None].repeat(3, axis=1)
    C0 = 0.28209479177387814
    n_sh = (a.sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(s, dtype=torch.float32, device=dev)),
        "quats": torch.nn.Parameter(torch.tensor(
            np.tile([1.0, 0, 0, 0], (len(xyz), 1)), dtype=torch.float32, device=dev)),
        "opacities": torch.nn.Parameter(torch.full(
            (len(xyz),), math.log(0.1 / 0.9), device=dev)),
        "sh0": torch.nn.Parameter(torch.tensor(
            (rgb - 0.5) / C0, dtype=torch.float32, device=dev)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(
            len(xyz), n_sh - 1, 3, device=dev)),
    }).to(dev)

    # INRIA learning rates; positions scale with the scene, everything else does not
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": v, "name": k}],
                                eps=1e-15, betas=(0.9, 0.999))
            for k, v in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opts["means"], gamma=(0.01) ** (1.0 / a.iters))

    # DefaultStrategy's defaults (refine 500-15000, reset every 3000) assume the
    # reference 30k schedule. On a longer run they would stop densifying a third
    # of the way in, so they scale with the schedule.
    strategy = DefaultStrategy(
        verbose=False,
        refine_stop_iter=max(15000, a.iters // 2),
        reset_every=max(3000, a.iters // 10),
    )
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    gt_cpu = torch.from_numpy(imgs)
    # True where the loss should ignore the pixel.
    ign_cpu = torch.from_numpy(masks) if masks is not None else None

    def supervised(pred, gt, i):
        """Blank out the ignored pixels by substituting the target into them.

        Both loss terms then see zero error there, and because `gt` is a
        constant the gradient through those pixels is exactly zero. Doing it
        this way rather than masking the L1 alone matters for SSIM, which is
        computed over an 11x11 window and would otherwise carry the object's
        error out into the pixels around it.
        """
        if ign_cpu is None:
            return pred
        keep = (~ign_cpu[i]).to(pred.device)[None].float()
        return pred * keep + gt * (1.0 - keep)
    Kt = torch.from_numpy(K).to(dev)[None]
    w2ct = torch.from_numpy(w2c).to(dev)
    win = gaussian_window(11, 1.5, dev)

    def render(idx, sh_deg):
        return rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=w2ct[idx][None], Ks=Kt, width=W, height=H,
            sh_degree=sh_deg, packed=False)

    def evaluate(sh_deg):
        """Held-out PSNR and SSIM.

        PSNR alone rewards getting the average brightness right, which a blurry
        render does well; SSIM is what moves when structure is actually
        recovered, so both are reported side by side.
        """
        tot_p = tot_s = 0.0
        with torch.no_grad():
            for i in test_idx:
                out, _, _ = render(int(i), sh_deg)
                gt = gt_cpu[i].to(dev).float() / 255.0
                pred = supervised(out[0].clamp(0, 1).permute(2, 0, 1),
                                  gt.permute(2, 0, 1), int(i)).permute(1, 2, 0)
                mse = torch.mean((pred - gt) ** 2).item()
                tot_p += 10 * math.log10(1.0 / max(mse, 1e-10))
                tot_s += float(ssim(pred.permute(2, 0, 1), gt.permute(2, 0, 1), win))
        n = len(test_idx)
        return tot_p / n, tot_s / n

    _p0, _s0 = evaluate(0)
    print(f"[eval] before training: PSNR {_p0:.2f} dB   SSIM {_s0:.4f}", flush=True)

    limit = a.vram_limit_gb * 1e9
    t0 = time.time()
    order = np.random.permutation(train_idx)
    ptr = 0
    for step in range(1, a.iters + 1):
        if ptr >= len(order):
            order = np.random.permutation(train_idx)
            ptr = 0
        i = int(order[ptr]); ptr += 1
        sh_deg = min(a.sh_degree, step // (a.iters // (a.sh_degree + 1) + 1))

        out, _, info = render(i, sh_deg)
        strategy.step_pre_backward(params=params, optimizers=opts, state=state,
                                   step=step, info=info)
        gt = (gt_cpu[i].to(dev).float() / 255.0).permute(2, 0, 1)
        pred = supervised(out[0].clamp(0, 1).permute(2, 0, 1), gt, i)
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * (1.0 - ssim(pred, gt, win))
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        strategy.step_post_backward(params=params, optimizers=opts, state=state,
                                    step=step, info=info, packed=False)
        for o in opts.values():
            o.step()
        sched.step()

        if torch.cuda.max_memory_allocated() > limit:
            print(f"[abort] VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB "
                  f"exceeded the {a.vram_limit_gb:.0f} GB limit at step {step}")
            break

        if step % 1000 == 0 or step == 1:
            print(f"  iter {step:6d}/{a.iters}  loss {loss.item():.4f}  "
                  f"gaussians {len(params['means']):,}  sh {sh_deg}  "
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB  "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
        if step % 7000 == 0:
            _p, _s = evaluate(sh_deg)
            print(f"  [eval] step {step}: PSNR {_p:.2f} dB   SSIM {_s:.4f}",
                  flush=True)

    final_p, final_s = evaluate(a.sh_degree)
    print(f"\n[eval] after training: PSNR {final_p:.2f} dB   SSIM {final_s:.4f}")
    print(f"[done] {len(params['means']):,} gaussians in "
          f"{(time.time()-t0)/60:.1f} min")
    save_ply(a.out, params)
    print(f"[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
