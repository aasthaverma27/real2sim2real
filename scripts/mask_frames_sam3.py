#!/usr/bin/env python3
"""Mask named objects out of every training frame, headlessly, via SAM3 text prompts.

The GUI route to per-frame masks is SAM3's *video* mode, which propagates from
frame 0 and therefore needs the objects clearly visible there - the constraint
that forced an earlier capture to be trimmed by 24 s, and that leaves frames
before the chosen start with no masks at all. SAM3 also ships a text-prompted
image path (`third_party/sam3/inference_image.py`), which has neither problem:
each frame is segmented independently, so frame ordering is irrelevant and
every frame gets a mask.

That matters here because the training frames come from 16 separate videos.
Hand-masking two of them in the GUI would have covered 157 of 450 frames; this
covers all of them, and costs no interactive work.

Output is one binary PNG per input frame - the union of every prompt - written
under `<out>/<frame stem>.png`, white where an object was found. `train_gs.py`
reads these and drops those pixels from the loss, so the splat reconstructs the
bare table from the views where each object is not in the way, rather than
baking a second copy of the apple and basket into the background.

Masks are dilated by default: a mask that hugs the silhouette exactly leaves a
rim of object-coloured pixels that the optimiser then paints onto the table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompts", required=True,
                   help='Comma separated, e.g. "red apple, brown woven basket"')
    p.add_argument("--checkpoint", type=Path,
                   default=Path(str(ROOT / "checkpoints" / "sam3" / "sam3.pt")))
    p.add_argument("--sam3-dir", type=Path,
                   default=Path(str(ROOT / "third_party" / "sam3")))
    p.add_argument("--min-score", type=float, default=0.35,
                   help="Reject detections below this confidence.")
    p.add_argument("--dilate", type=int, default=12,
                   help="Pixels to grow each mask. A tight mask leaves an "
                        "object-coloured rim that the optimiser paints onto "
                        "the table.")
    p.add_argument("--max-frac", type=float, default=0.45,
                   help="Reject a detection covering more than this fraction "
                        "of the frame - a text prompt that latches onto the "
                        "whole table would erase the scene.")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    sys.path.insert(0, str(a.sam3_dir))
    import cv2
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    prompts = [s.strip() for s in a.prompts.split(",") if s.strip()]
    frames = sorted(list(a.images.glob("*.jpg")) + list(a.images.glob("*.png")))
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        raise SystemExit(f"no images in {a.images}")
    print(f"[mask] {len(frames)} frames, prompts {prompts}", flush=True)

    model = build_sam3_image_model(checkpoint_path=str(a.checkpoint))
    proc = Sam3Processor(model)
    a.out.mkdir(parents=True, exist_ok=True)

    kern = (np.ones((a.dilate * 2 + 1,) * 2, np.uint8) if a.dilate > 0 else None)
    stats = {q: 0 for q in prompts}
    covered = []
    for i, f in enumerate(frames):
        img = Image.open(f).convert("RGB")
        W, H = img.size
        state = proc.set_image(img)
        acc = np.zeros((H, W), np.uint8)
        for q in prompts:
            proc.reset_all_prompts(state)
            proc.set_text_prompt(prompt=q, state=state)
            masks = state.get("masks")
            scores = state.get("scores")
            if masks is None or masks.shape[0] == 0:
                continue
            s = scores.squeeze(-1) if scores.ndim > 1 else scores
            keep = (s >= a.min_score).nonzero().flatten().tolist()
            for k in keep:
                m = masks[k, 0].detach().cpu().numpy().astype(np.uint8)
                if m.mean() > a.max_frac:      # latched onto the whole scene
                    continue
                acc |= m
                stats[q] += 1
        if kern is not None and acc.any():
            acc = cv2.dilate(acc, kern)
        cv2.imwrite(str(a.out / f"{f.stem}.png"), acc * 255)
        covered.append(float(acc.mean()))
        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(f"  {i+1}/{len(frames)}  mean coverage "
                  f"{100*np.mean(covered):.1f}%", flush=True)
        del state
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    c = np.array(covered)
    print(f"\n[mask] wrote {len(frames)} masks -> {a.out}")
    print(f"[mask] coverage: mean {100*c.mean():.1f}%, "
          f"max {100*c.max():.1f}%, empty on {int((c==0).sum())} frames")
    for q, n in stats.items():
        print(f"[mask]   '{q}' detected in {n} frames")
    if (c == 0).sum() > len(frames) * 0.3:
        print("[mask] WARNING: over 30% of frames got no mask at all - "
              "check the prompt wording against the images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
