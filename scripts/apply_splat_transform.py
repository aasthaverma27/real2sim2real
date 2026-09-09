#!/usr/bin/env python3
"""Bake a similarity transform into a splat PLY.

A 3D gaussian transforms exactly under a similarity: rotate the mean and scale
it, multiply the rotation quaternion, and add log(s) to the log-scales. What
does NOT transform trivially is the spherical-harmonic colour: bands above DC
encode direction-dependent colour in the *world* frame, so rotating the splat
requires rotating the SH coefficients by the corresponding Wigner D-matrices.

This drops them instead, keeping DC only, and says so rather than hiding it.
The reason is that the scene's existing background (bg_lm.ply) is already
DC-only, so MuGS renders both identically, and an indoor diffuse room loses
little to view-independent colour. Getting degree-3 rotation subtly wrong would
cost more than the bands are worth. If the loss ever matters, the fix is to
implement the D-matrices, not to skip the rotation.

    scripts/apply_splat_transform.py --ply in.ply --out out.ply \
        --scale 0.41 --yaw 273.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--poses", type=Path,
                   default=Path("outputs/g1_ok/roommesh/colmap_poses.json"))
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--table-h", type=float, default=-1.1550)
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.648, 0.676])
    p.add_argument("--world-table", type=float, nargs=3, default=[-0.055, 0.724, 0.713])
    p.add_argument("--min-opacity", type=float, default=0.01,
                   help="drop near-invisible gaussians; they cost memory and "
                        "contribute nothing")
    a = p.parse_args()

    import mujoco
    from plyfile import PlyData, PlyElement

    def yaw_mat(deg):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def basis_from_up(up):
        up = up / np.linalg.norm(up)
        ax = np.cross(up, [0.0, 0.0, 1.0])
        n = np.linalg.norm(ax)
        if n < 1e-9:
            return np.eye(3)
        ax /= n
        th = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    ply = PlyData.read(str(a.ply))
    vert = ply["vertex"]
    names = list(vert.data.dtype.names)
    n0 = len(vert)
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)

    poses = json.load(open(a.poses))
    C = np.array([poses[k]["C"] for k in poses])
    up = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
    if (xyz[::97] @ up < np.median(C @ up)).mean() < 0.5:
        up = -up
    R0 = basis_from_up(up)
    b1 = np.cross(up, [0.0, 0.0, 1.0]); b1 /= np.linalg.norm(b1)
    b2 = np.cross(up, b1)
    anchor = a.table_xy[0] * b1 + a.table_xy[1] * b2 + a.table_h * up
    R = yaw_mat(a.yaw) @ R0
    s = a.scale
    t = np.asarray(a.world_table, float) - s * (R @ anchor)
    print(f"[apply] scale {s:.4f}  yaw {a.yaw:.2f} deg  det(R) {np.linalg.det(R):+.6f}")
    print(f"[apply] t {np.round(t,4)}")

    opa = 1.0 / (1.0 + np.exp(-vert["opacity"].astype(np.float32)))
    keep = opa > a.min_opacity
    print(f"[apply] keeping {keep.sum():,} of {n0:,} "
          f"(dropped {n0-keep.sum():,} below opacity {a.min_opacity})")

    out_names = [n for n in names if not n.startswith("f_rest")]
    dropped = len(names) - len(out_names)
    if dropped:
        print(f"[apply] dropping {dropped} spherical-harmonic bands, keeping DC "
              f"(see the module docstring)")
    dt = [(n, "f4") for n in out_names]
    arr = np.empty(int(keep.sum()), dtype=dt)
    for n in out_names:
        arr[n] = vert[n][keep]

    # means: full similarity
    new_xyz = s * (xyz[keep] @ R.T) + t
    arr["x"], arr["y"], arr["z"] = (new_xyz[:, 0].astype(np.float32),
                                    new_xyz[:, 1].astype(np.float32),
                                    new_xyz[:, 2].astype(np.float32))
    # log-scales: log(s * exp(l)) = l + log(s)
    for i in range(3):
        arr[f"scale_{i}"] = (vert[f"scale_{i}"][keep] + np.log(s)).astype(np.float32)
    # rotation: compose quaternions
    q = np.stack([vert[f"rot_{i}"][keep] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qR = np.empty(4)
    mujoco.mju_mat2Quat(qR, R.reshape(9))
    out = np.empty_like(q)
    tmp = np.empty(4)
    for i in range(len(q)):
        mujoco.mju_mulQuat(tmp, qR, q[i])
        out[i] = tmp
    for i in range(4):
        arr[f"rot_{i}"] = out[:, i].astype(np.float32)
    # normals are unused by every renderer here, but keep them consistent
    if "nx" in out_names:
        nrm = np.stack([vert["nx"][keep], vert["ny"][keep], vert["nz"][keep]], 1)
        nrm = nrm.astype(np.float64) @ R.T
        arr["nx"], arr["ny"], arr["nz"] = (nrm[:, 0].astype(np.float32),
                                           nrm[:, 1].astype(np.float32),
                                           nrm[:, 2].astype(np.float32))

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(a.out))
    print(f"[apply] wrote {a.out}")
    print(f"[apply] world bbox x {new_xyz[:,0].min():+.2f}..{new_xyz[:,0].max():+.2f}"
          f"  y {new_xyz[:,1].min():+.2f}..{new_xyz[:,1].max():+.2f}"
          f"  z {new_xyz[:,2].min():+.2f}..{new_xyz[:,2].max():+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
