#!/usr/bin/env python3
"""Set the background splat's yaw by eye, against the photograph.

Yaw is the last free parameter: gravity comes from the camera orbit, scale and
position are pinned by the table. Five automated objectives gave five different
answers (F14) - full-frame NCC 340.25, wall NCC 92.50, apple position 273.25,
depth agreement 309.25, window brightness 353.75 - because each locks onto a
feature that repeats or mirrors somewhere else in the room. The only reliable
automatic discriminator was the mirror test: the apple sits left of the basket
in the photograph, which is true only from the correct side of the room, and
that brackets the answer between 273 and 310 degrees.

Inside that band a person reading the picture beats every metric tried, so this
exposes exactly one knob. The splat stays put and the CAMERA is inverse-
transformed, which is why it re-renders instantly instead of moving 2.4M
gaussians per keypress.

Keys
  LEFT / RIGHT   yaw -/+
  F              coarse (2 deg) / fine (0.25 deg)
  O              cycle blend: 50/50 -> photo only -> splat only
  ENTER          print and save the chosen yaw
  ESC            quit without saving
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def yaw_mat(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def basis_from_up(up: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up)
    ax = np.cross(up, [0.0, 0.0, 1.0])
    n = np.linalg.norm(ax)
    if n < 1e-9:
        return np.eye(3)
    ax /= n
    th = np.arccos(np.clip(up @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, default=None,
                   help="fills in every path below from a build_sim_scene.sh "
                        "scene directory")
    p.add_argument("--ply", type=Path, default=None, help="<scene>/gs_wide.ply")
    p.add_argument("--poses", type=Path, default=None,
                   help="<scene>/roommesh/colmap_poses.json")
    p.add_argument("--photo", type=Path, default=None, help="<scene>/inputs/frame0.png")
    p.add_argument("--scene", type=Path, default=None,
                   help="<scene>/mujoco/scene_scaled.xml")
    p.add_argument("--out", type=Path, default=None, help="<scene>/gs_yaw_manual.json")
    p.add_argument("--scale", type=float, default=0.5075)
    p.add_argument("--yaw", type=float, default=309.25, help="starting yaw")
    p.add_argument("--table-h", type=float, default=-1.1550)
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.648, 0.676])
    p.add_argument("--world-table", type=float, nargs=3, default=[-0.055, 0.724, 0.713])
    p.add_argument("--height", type=int, default=820)
    p.add_argument("--opacity", type=float, default=0.03)
    p.add_argument("--pos", type=int, nargs=2, default=[30, 40])
    a = p.parse_args()
    sd = a.scene_dir
    if sd is not None:
        a.ply = a.ply or sd / "gs_wide.ply"
        a.poses = a.poses or sd / "roommesh" / "colmap_poses.json"
        a.photo = a.photo or sd / "inputs" / "frame0.png"
        a.scene = a.scene or sd / "mujoco" / "scene_scaled.xml"
        a.out = a.out or sd / "gs_yaw_manual.json"
    missing = [k for k in ("ply", "poses", "photo", "scene", "out")
               if getattr(a, k) is None]
    if missing:
        p.error("pass --scene-dir, or all of: " + ", ".join("--" + m for m in missing))

    import cv2
    import mujoco
    import torch
    from gsplat import rasterization
    from plyfile import PlyData

    dev = "cuda"
    v = PlyData.read(str(a.ply))["vertex"]
    keep = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))) > a.opacity
    means = np.stack([v["x"], v["y"], v["z"]], 1)[keep].astype(np.float32)
    T = [torch.from_numpy(x).to(dev) for x in (
        means,
        np.stack([v[f"rot_{i}"] for i in range(4)], 1)[keep].astype(np.float32),
        np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)[keep]).astype(np.float32),
        (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32))))[keep].astype(np.float32),
        np.clip(0.5 + 0.2820948 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[keep],
                0, 1).astype(np.float32))]
    print(f"[yaw] {len(means):,} gaussians on the GPU", flush=True)

    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cap")
    if ci < 0:
        raise SystemExit(
            f"{a.scene} has no camera named 'cap' (the pose the object photo was "
            "taken from). Add one at that pose - <camera name=\"cap\" pos=... "
            "xyaxes=... fovy=.../> in the worldbody - or pass --scene with a "
            "scene that has it.")
    Ccap = data.cam_xpos[ci].copy()
    Rocv = data.cam_xmat[ci].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    fovy = float(model.cam_fovy[ci])

    photo = cv2.imread(str(a.photo))
    H = a.height
    W = int(round(H * photo.shape[1] / photo.shape[0]))
    photo = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)
    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]],
                     dtype=torch.float32, device=dev)

    poses = json.load(open(a.poses))
    C = np.array([poses[k]["C"] for k in poses])
    up = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][2]
    if (means[::97] @ up < np.median(C @ up)).mean() < 0.5:
        up = -up
    R0 = basis_from_up(up)
    b1 = np.cross(up, [0.0, 0.0, 1.0]); b1 /= np.linalg.norm(b1)
    b2 = np.cross(up, b1)
    anchor = a.table_xy[0] * b1 + a.table_xy[1] * b2 + a.table_h * up
    wt = np.asarray(a.world_table, float)

    def render(yaw):
        R = yaw_mat(yaw) @ R0
        t = wt - a.scale * (R @ anchor)
        Cs = (R.T / a.scale) @ (Ccap - t)
        Rcs = R.T @ Rocv
        w2c = np.eye(4)
        w2c[:3, :3] = Rcs.T
        w2c[:3, 3] = -Rcs.T @ Cs
        vm = torch.from_numpy(w2c.astype(np.float32))[None].to(dev)
        with torch.no_grad():
            img, _, _ = rasterization(*T, vm, K, W, H, packed=False)
        return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]

    win = "splat yaw - match the room to its own photograph"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.moveWindow(win, a.pos[0], a.pos[1])
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1.0)
    except Exception:
        pass
    print(__doc__.split("Keys")[1], flush=True)

    yaw, fine, blend = a.yaw, False, 1
    while True:
        r = render(yaw)
        if blend == 1:
            img = cv2.addWeighted(photo, 0.5, r, 0.5, 0.0)
        elif blend == 2:
            img = photo.copy()
        else:
            img = r.copy()
        step = 0.25 if fine else 2.0
        for i, line in enumerate([f"yaw {yaw:7.2f} deg   step {step}   "
                                  f"scale {a.scale}",
                                  "LEFT/RIGHT yaw   F fine   O blend   "
                                  "ENTER save   ESC quit",
                                  "the apple must be LEFT of the basket - "
                                  "otherwise you are on the wrong side"]):
            cv2.putText(img, line, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(img, line, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            print("quit without saving")
            break
        elif k == 81:
            yaw = (yaw - step) % 360.0
        elif k == 83:
            yaw = (yaw + step) % 360.0
        elif k in (ord("f"), ord("F")):
            fine = not fine
        elif k in (ord("o"), ord("O")):
            blend = (blend + 1) % 3
        elif k in (13, 10):
            a.out.write_text(json.dumps(dict(yaw_deg=yaw, scale=a.scale), indent=1))
            print(f"\nsaved yaw {yaw:.2f} deg, scale {a.scale} -> {a.out}")
            print("bake it with:")
            print("re-run the build from the align stage with:")
            print(f"  --splat-scale {a.scale} --splat-yaw {yaw:.2f} --from align")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
