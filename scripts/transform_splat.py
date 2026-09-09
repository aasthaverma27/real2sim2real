#!/usr/bin/env python3
"""Apply a similarity transform to a Gaussian splat PLY.

A splat is not a point cloud: means move, but scales must be multiplied by the
similarity's scale factor and orientations must be composed with its rotation,
or the gaussians keep their old size and tilt in the new frame and the render
turns to streaks. Spherical harmonics above the DC term are direction-dependent
and would need rotating too; degree>0 coefficients are dropped instead, which
costs view-dependent shading but never introduces a wrong one.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--transform", type=Path, required=True, help="4x4 .npy")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-sh", action="store_true",
                   help="Keep higher-order SH unrotated (wrong, but preserves them).")
    a = p.parse_args()

    T = np.load(a.transform)
    M = T[:3, :3]
    s = float(np.mean(np.linalg.norm(M, axis=0)))
    R = M / s
    if abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise SystemExit(f"transform is not a similarity (det R = {np.linalg.det(R):.4f})")
    print(f"[xform] scale {s:.6f}, translation {np.round(T[:3,3],4)}")

    v = PlyData.read(str(a.ply))["vertex"]
    names = list(v.data.dtype.names)
    col = lambda *k: np.column_stack([np.asarray(v[n], dtype=np.float32) for n in k])
    means = col("x", "y", "z")
    quats = col("rot_0", "rot_1", "rot_2", "rot_3")
    quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    log_s = col("scale_0", "scale_1", "scale_2")
    print(f"[xform] {len(means):,} gaussians")

    means = (s * (R @ means.T).T + T[:3, 3]).astype(np.float32)
    log_s = (log_s + np.log(s)).astype(np.float32)
    quats = quat_mul(mat_to_quat(R), quats).astype(np.float32)
    quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)

    n_rest = len([n for n in names if n.startswith("f_rest_")])
    keep_rest = n_rest if a.keep_sh else 0
    if n_rest and not a.keep_sh:
        print(f"[xform] dropping {n_rest} higher-order SH coefficients "
              f"(they are direction-dependent and would need rotating)")

    cols = [("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4")]
    cols += [(f"f_dc_{i}","f4") for i in range(3)]
    cols += [(f"f_rest_{i}","f4") for i in range(keep_rest)]
    cols += [("opacity","f4")] + [(f"scale_{i}","f4") for i in range(3)]
    cols += [(f"rot_{i}","f4") for i in range(4)]
    arr = np.empty(len(means), dtype=cols)
    arr["x"], arr["y"], arr["z"] = means.T
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(3):
        arr[f"f_dc_{i}"] = np.asarray(v[f"f_dc_{i}"], dtype=np.float32)
    for i in range(keep_rest):
        arr[f"f_rest_{i}"] = np.asarray(v[f"f_rest_{i}"], dtype=np.float32)
    arr["opacity"] = np.asarray(v["opacity"], dtype=np.float32)
    for i in range(3):
        arr[f"scale_{i}"] = log_s[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(a.out))
    print(f"[xform] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
