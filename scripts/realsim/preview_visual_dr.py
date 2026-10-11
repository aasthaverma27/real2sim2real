#!/usr/bin/env python3
"""Contact sheet of visual-randomization samples, through the real renderer.

Renders the photoreal composite (splat background + MuJoCo meshes) from the
robot's own cameras for a range of seeds, exactly as vla_camera_server.py would
after a reset with that seed, and tiles them into one image. Look at it before
recording: no black or washed-out frames, the apple still red, the basket still
brown, the meshes not looking pasted onto the room.

    $RS_PY_SIM scripts/realsim/preview_visual_dr.py
    ... --seeds 1-24 --camera g1_head_pinhole --out outputs/visual_dr/preview.jpg

Robot at the scene's rest pose - lighting, not motion, is what is being checked.
Needs ~3 GB of VRAM for the splat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "MuGS" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--config", default=None, help="default configs/visual_randomization.yaml")
    p.add_argument("--seeds", default="0-15", help="e.g. 0-15 or 3,7,9 (0 = nominal)")
    p.add_argument("--camera", default="g1_head_pinhole")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--tile", type=int, default=320, help="tile width in the sheet")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--exclude-geoms", default="floor,ground,room")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "visual_dr" / "preview.jpg")
    a = p.parse_args()

    import cv2
    import mujoco
    from mugs.sensors import GaussianSensor, GaussianSensorConfig
    from visual_randomizer import VisualDRFollower

    mjcf = a.scene_dir / "mujoco" / "scene_g1.xml"
    ply = next(f for f in (a.scene_dir / "gs_bg_notripod.ply",
                           a.scene_dir / "gs_bg_clean.ply",
                           a.scene_dir / "gs_bg.ply") if f.is_file())
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    exclude = {x.strip() for x in a.exclude_geoms.split(",") if x.strip()}
    fg = [g for g in range(model.ngeom)
          if int(model.geom_bodyid[g]) != 0 and int(model.geom_group[g]) != 3
          and not any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                      for x in exclude)]
    sensor = GaussianSensor(GaussianSensorConfig(
        width=a.width, height=a.height, background_ply_path=str(ply),
        render_mode="hybrid", robot_geom_ids=fg, use_sh=True,
        depth_composite=True, camera_convention="mujoco"))
    sensor._get_renderer(model)
    sensor._scene_option.sitegroup[:] = 0

    follower = VisualDRFollower(model, a.config, [sensor], tag="preview")

    class Fake:                       # stands in for MirrorReceiver.visual
        visual = None

    tiles = []
    th = int(round(a.tile * a.height / a.width))
    for i, seed in enumerate(parse_seeds(a.seeds)):
        Fake.visual = (seed, i)
        follower.update(Fake)
        img = sensor.render(model, data, a.camera)[:, :, ::-1]
        img = cv2.resize(np.ascontiguousarray(img), (a.tile, th),
                         interpolation=cv2.INTER_AREA)
        pr = follower.params
        label = (f"seed {seed} nominal" if pr["nominal"] else
                 f"seed {seed} L{pr.get('total_diffuse', 0):.2f}")
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)

    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + a.cols])
                       for r in range(0, len(tiles), a.cols)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), sheet)
    print(f"[preview] {len(parse_seeds(a.seeds))} samples -> {a.out}")
    sensor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
