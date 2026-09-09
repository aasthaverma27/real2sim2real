#!/usr/bin/env python3
"""Run SAM3D multi-view inference and persist the object poses it computes.

run_inference.py computes each object's pose (rotation/translation/scale,
local->camera of view 0) but returns nothing and never writes it; its own
compose_scene_from_results() is defined and never called. So the good
multi-view meshes end up with no placement.

This wraps the repo's pipeline without editing it: it intercepts
InferencePipeline.run_multi_view, records the pose alongside each object, and
then composes the scene with the repo's own compose function.

    python scripts/capture_multiview_poses.py --input-path outputs/robot_video/multi_mask \
        --config checkpoints/sam-3d-objects/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-path", type=Path, required=True, help="multi_mask directory")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, help="composed scene glb")
    a = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    sam3d = root / "third_party" / "sam-3d-objects"
    sys.path.insert(0, str(sam3d))
    sys.path.insert(0, str(sam3d / "sam3d_objects"))

    import torch
    import run_inference as ri
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline

    captured: list[dict] = []
    current = {"path": None}

    orig_multi = InferencePipeline.run_multi_view

    def patched_multi(self, *args, **kwargs):
        out = orig_multi(self, *args, **kwargs)
        rec = {"object": current["path"]}
        for key in ("rotation", "translation", "scale"):
            if key in out:
                v = out[key]
                rec[key] = (v.detach().float().cpu().numpy().tolist()
                            if isinstance(v, torch.Tensor) else np.asarray(v).tolist())
        captured.append(rec)
        print(f"[poses] captured for {current['path']}: "
              f"{[k for k in ('rotation','translation','scale') if k in rec]}")
        return out

    InferencePipeline.run_multi_view = patched_multi

    orig_run = ri.run_inference

    def patched_run(input_path, *args, **kwargs):
        current["path"] = Path(input_path).parent.name
        return orig_run(input_path, *args, **kwargs)

    ri.run_inference = patched_run

    sys.argv = ["run_inference.py", "--input_path", str(a.input_path),
                "--config", str(a.config)]
    ri.main()

    out_json = a.input_path / "multiview_poses.json"
    out_json.write_text(json.dumps(captured, indent=2), encoding="utf-8")
    print(f"[poses] wrote {out_json} ({len(captured)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
