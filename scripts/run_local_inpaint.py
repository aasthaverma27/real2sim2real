#!/usr/bin/env python3
"""Local background-completion adapter (no external API).

Implements the same CLI contract as scripts/run_gemini_image_edit.py:
    --image {image} --mask {mask} --prompt {prompt} --output {output} --status {status}

Fallback for when the Gemini image model is unavailable (the free API tier
reports `limit: 0` for gemini-*-flash-image). Uses OpenCV Navier-Stokes
inpainting at reduced scale, then refines at full resolution. This is
geometric infill, not semantic generation: it reconstructs plausible
surface texture but cannot invent occluded structure the way the paid
image model does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def inpaint(image_path: Path, mask_path: Path, output_path: Path) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot read mask: {mask_path}")
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.uint8) * 255
    # dilate slightly so object edge halos are covered
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)

    # coarse pass at half scale fills large holes with less smearing
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(binary, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    coarse = cv2.inpaint(small, small_mask, 7, cv2.INPAINT_NS)
    coarse_up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    seeded = img.copy()
    seeded[binary > 0] = coarse_up[binary > 0]
    result = cv2.inpaint(seeded, binary, 3, cv2.INPAINT_TELEA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return {"filled_pixels": int((binary > 0).sum()), "size": [w, h]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--prompt", default="", help="Accepted for contract compatibility; unused.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--status", type=Path)
    args = p.parse_args()

    info = inpaint(args.image, args.mask, args.output)
    status = {
        "status": "local_inpaint_complete",
        "provider": "opencv-ns-telea",
        "image": str(args.image),
        "mask": str(args.mask),
        "output": str(args.output),
        "output_mode": "geometric_infill",
        "masked_pixels": info["filled_pixels"],
        "note": "local fallback; semantic quality below the Gemini image model",
    }
    status_path = args.status or args.output.with_name(args.output.stem + "_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
