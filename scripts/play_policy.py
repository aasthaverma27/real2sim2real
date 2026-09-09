#!/usr/bin/env python3
"""Watch a policy act in the photoreal scene, live.

mjlab's own viewer draws MuJoCo's world, and the Gaussian splat is not in
MuJoCo's world - it exists only inside the camera sensor. So watching a policy
"with the background" means displaying the sensor's own output, which is what this
does: one environment, the composite rendered every control step, in a window.

Works with no checkpoint at all, so the loop can be seen before there is anything
trained: --policy sine sweeps the arm, --policy random flails, --policy zero holds.

    bash scripts/run_play_policy.sh                       # sine sweep, no checkpoint
    bash scripts/run_play_policy.sh --checkpoint model.pt # a trained policy

Keys:  SPACE pause/run   R reset   B cycle composite/background/foreground
       ESC or Q quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_policy(path: Path, device):
    """Accept a TorchScript module, an nn.Module, or a dict holding one."""
    import torch

    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for key in ("policy", "model", "actor", "state_dict"):
            if key in obj and hasattr(obj[key], "__call__"):
                obj = obj[key]
                break
        else:
            raise SystemExit(
                f"{path} is a dict of tensors ({list(obj)[:4]}...), not a callable "
                f"policy. Save the module itself, or a TorchScript trace.")
    obj.eval() if hasattr(obj, "eval") else None
    print(f"[play] loaded policy {type(obj).__name__} from {path.name}")
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, default=ROOT / "outputs" / "video_scene")
    p.add_argument("--mjcf", type=Path, help="default: <scene>/mujoco/scene_batch.xml")
    p.add_argument("--camera", default="capture",
                   help="comma-separated. 'capture' is a third-person view; "
                        "'g1_ego_view' is what the robot itself sees - the image a "
                        "policy trained on that camera actually receives.")
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--policy", default="sine",
                   choices=["sine", "random", "zero"],
                   help="used when no --checkpoint is given")
    p.add_argument("--link", default="g1_right_wrist_yaw_link")
    p.add_argument("--target", default="obj3_blue_octopus_plush_toy")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--show", default="cycle",
                   choices=["cycle", "all", "rgb", "background", "foreground"],
                   help="'all' shows MuJoCo-only, 3DGS-only and the hybrid side by "
                        "side at once - the same three panels as the MuGS figure, "
                        "but live. 'cycle' shows one at a time, B to switch.")
    p.add_argument("--record", type=Path, metavar="MP4",
                   help="also write what is displayed to a video")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    import cv2
    import torch

    from robosnap.envs import BatchedPhotorealEnv, EnvConfig, ReachTask

    cams = [s.strip() for s in a.camera.split(",") if s.strip()]
    cfg = EnvConfig(scene_dir=a.scene_dir, mjcf=a.mjcf, num_worlds=1,
                    cameras=cams, width=a.width, height=a.height,
                    horizon=a.horizon, randomize_objects=0.10)
    env = BatchedPhotorealEnv(cfg, ReachTask(a.link, a.target))
    if env.num_actions == 0:
        raise SystemExit(
            f"{env.cfg.mjcf or 'scene'} has no actuators - nothing for a policy to "
            f"drive. Use a scene with a robot (mujoco/scene_batch.xml).")
    print(f"[play] {env.num_actions} actions, cameras {cams}, "
          f"{a.width}x{a.height}")

    policy = load_policy(a.checkpoint, cfg.device) if a.checkpoint else None
    if policy is None:
        print(f"[play] no checkpoint - driving with '{a.policy}'")

    writer = None
    if a.record:
        n_panel = 3 if a.show == "all" else len(cams)
        writer = cv2.VideoWriter(str(a.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (a.width * n_panel, a.height))

    titles = {"rgb": "3 - HYBRID  (policy plays here)",
              "background": "2 - 3DGS ONLY  (your reconstructed room)",
              "foreground": "1b - MuJoCo geometry only",
              "all": "MuJoCo | 3DGS | hybrid", "cycle": "policy in the photoreal scene"}
    win = f"RoboSnap  {titles.get(a.show, a.show)}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    panels_wide = 3 if a.show == "all" else len(cams)
    cv2.resizeWindow(win, min(1600, a.width * panels_wide), min(980, a.height))
    layers = ["rgb", "background", "foreground"]
    layer = layers.index(a.show) if a.show in layers else 0
    gen = torch.Generator(device=cfg.device).manual_seed(0)

    obs = env.reset()
    running, step, t0, frames = True, 0, time.perf_counter(), 0
    reward = torch.zeros(1, device=cfg.device)
    info = {"distance": obs["task_distance"], "success": torch.zeros(1, dtype=torch.bool)}
    while True:
        if running:
            if policy is not None:
                with torch.no_grad():
                    flat = torch.cat([obs["qpos"], obs["task_delta"]], dim=1)
                    act = policy(flat)
                    if isinstance(act, (tuple, list)):
                        act = act[0]
                    act = act[:, :env.num_actions].to(cfg.device).float()
            elif a.policy == "sine":
                phase = step * 0.05
                act = torch.sin(phase + torch.arange(
                    env.num_actions, device=cfg.device) * 0.7).unsqueeze(0)
            elif a.policy == "random":
                act = torch.rand((1, env.num_actions), generator=gen,
                                 device=cfg.device) * 2 - 1
            else:
                act = torch.zeros((1, env.num_actions), device=cfg.device)
            obs, reward, done, info = env.step(act)
            step += 1

        comp = env.render(components=True)
        if a.show == "all":
            panels = [("MuJoCo only", "foreground"), ("3DGS only", "background"),
                      ("hybrid  <- the policy view", "rgb")]
            views = []
            for title, key in panels:
                v = np.ascontiguousarray(comp[key][0, 0].cpu().numpy())
                cv2.rectangle(v, (0, v.shape[0] - 26), (v.shape[1], v.shape[0]),
                              (16, 16, 20), -1)
                col = ((120, 240, 255) if key == "rgb" else (200, 200, 205))
                cv2.putText(v, title, (8, v.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
                views.append(v)
            img = np.ascontiguousarray(np.hstack(views))
            frames += 1
            fps = frames / max(time.perf_counter() - t0, 1e-6)
            cv2.rectangle(img, (0, 0), (img.shape[1], 52), (16, 16, 20), -1)
            cv2.putText(img, f"step {step}   {fps:.1f} fps   camera {cams[0]}",
                        (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 214, 102),
                        1, cv2.LINE_AA)
            cv2.putText(img, f"reward {float(reward.mean()):+.3f}   "
                             f"dist {float(info['distance'].mean()):.3f} m   "
                             f"success {'yes' if bool(info['success'][0]) else 'no'}",
                        (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240),
                        1, cv2.LINE_AA)
            cv2.imshow(win, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if writer is not None:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                running = not running
            elif k in (ord("r"), ord("R")):
                obs = env.reset(); step = 0
            continue

        views = []
        for ci, cname in enumerate(cams):
            v = comp[layers[layer]][ci, 0].cpu().numpy()
            if v.shape[-1] == 1:
                v = np.repeat(v, 3, axis=-1)
            v = np.ascontiguousarray(v)
            if len(cams) > 1:
                cv2.rectangle(v, (0, v.shape[0] - 26), (v.shape[1], v.shape[0]),
                              (16, 16, 20), -1)
                tag = ("ROBOT'S OWN CAMERA" if "ego" in cname or "mono" in cname
                       else "third-person view")
                cv2.putText(v, f"{cname}  -  {tag}", (8, v.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 240, 150), 1,
                            cv2.LINE_AA)
            views.append(v)
        img = np.ascontiguousarray(np.hstack(views))
        frames += 1
        fps = frames / max(time.perf_counter() - t0, 1e-6)
        bar = f"{layers[layer]}  step {step}  {fps:.1f} fps"
        stat = (f"reward {float(reward.mean()):+.3f}   "
                f"dist {float(info['distance'].mean()):.3f} m   "
                f"success {'yes' if bool(info['success'][0]) else 'no'}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 52), (16, 16, 20), -1)
        cv2.putText(img, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 214, 102), 1, cv2.LINE_AA)
        cv2.putText(img, stat, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (235, 235, 240), 1, cv2.LINE_AA)
        cv2.imshow(win, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if writer is not None:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            running = not running
        elif k in (ord("r"), ord("R")):
            obs = env.reset()
            step = 0
        elif k in (ord("b"), ord("B")):
            layer = (layer + 1) % len(layers)

    if writer is not None:
        writer.release()
        print(f"[play] wrote {a.record}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
