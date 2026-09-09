#!/usr/bin/env python3
"""Estimate physical properties for reconstructed objects.

A photo gives geometry, not physics. Mass, density and friction have to come
from somewhere else - this fills them in automatically instead of by hand.

Two modes:
  * lookup  (default, offline) - match object names against a material table
  * vlm     - ask Gemini for material/density/friction per object

Writes <scene-dir>/object_properties.json, consumed by
scripts/export_mujoco.py --materials.

    python scripts/estimate_properties.py --scene-dir outputs/robot_cube_03
    python scripts/estimate_properties.py --scene-dir outputs/... --mode vlm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# (substring, density kg/m^3, sliding friction) - first match wins.
MATERIAL_TABLE: list[tuple[str, float, float]] = [
    # Ordered by priority: specific object types first, generic materials last,
    # and FIRST match wins. "humanoid robot sitting at the table" must resolve to
    # `robot`, and "brown plastic woven basket" to `basket` (mostly air), not
    # `plastic`. Densities are EFFECTIVE bulk values for the whole object.
    ("plush",      35,   0.9),
    ("stuffed",    35,   0.9),
    ("foam",       60,   0.9),
    ("cloth",     150,   0.8),
    ("robot",    1200,   0.6),
    ("basket",    250,   0.7),
    ("crate",     300,   0.7),
    ("rubik",     600,   0.6),
    ("bottle",    150,   0.5),
    ("mug",       800,   0.6),
    ("cup",       300,   0.6),
    ("bowl",      600,   0.6),
    ("toy",       200,   0.8),
    ("cardboard", 250,   0.7),
    ("paper",     400,   0.7),
    ("block",     600,   0.6),
    ("cube",      600,   0.6),
    ("table",     650,   0.6),
    ("desk",      650,   0.6),
    ("shelf",     650,   0.6),
    ("wood",      650,   0.6),
    ("plastic",   900,   0.5),
    ("glass",    2500,   0.4),
    ("ceramic",  2300,   0.5),
    ("steel",    7800,   0.4),
    ("metal",    2700,   0.4),
]
DEFAULT_DENSITY, DEFAULT_FRICTION = 400.0, 0.6

VLM_PROMPT = """You are given a list of objects detected in a real photograph.
For each object return its most likely physical properties as JSON:

{"objects":[{"name":"<echo the name>","material":"<e.g. plastic, wood, plush>",
"density_kg_m3":<number>,"friction":<0.1-1.5>}]}

Use EFFECTIVE bulk density of the whole object, not the raw material density:
a woven basket or a hollow bottle is mostly air, so its effective density is far
below solid plastic. A plush toy is roughly 30-60. Solid wood ~650. Return only JSON.

Objects:
"""


def lookup(name: str) -> tuple[str, float, float]:
    """First match wins, using the priority ordering of MATERIAL_TABLE."""
    n = name.lower()
    for key, dens, fric in MATERIAL_TABLE:
        if key in n:
            return key, float(dens), float(fric)
    return "unknown", DEFAULT_DENSITY, DEFAULT_FRICTION


def load_names(scene_dir: Path) -> list[str]:
    f = scene_dir / "object.txt"
    if not f.is_file():
        raise SystemExit(f"missing {f}")
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def from_vlm(names: list[str], scene_dir: Path) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[props] no GEMINI_API_KEY; falling back to lookup", file=sys.stderr)
        return None
    root = Path(__file__).resolve().parents[1]
    img = scene_dir / "image.png"
    out = scene_dir / "_vlm_props.json"
    prompt = VLM_PROMPT + "\n".join(f"- {n}" for n in names)
    pfile = scene_dir / "_vlm_props_prompt.txt"
    pfile.write_text(prompt, encoding="utf-8")
    cmd = [os.environ.get("PY_ASSET", sys.executable),
           str(root / "scripts" / "run_gemini_vlm.py"),
           "--image", str(img), "--prompt", str(pfile), "--output", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if r.returncode != 0 or not out.is_file():
            print(f"[props] VLM call failed; falling back to lookup", file=sys.stderr)
            return None
        return json.loads(out.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[props] VLM error {e}; falling back to lookup", file=sys.stderr)
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("lookup", "vlm"), default="lookup")
    p.add_argument("--output", type=Path)
    a = p.parse_args()

    names = load_names(a.scene_dir)
    vlm = from_vlm(names, a.scene_dir) if a.mode == "vlm" else None
    vlm_map = {}
    if vlm:
        for o in vlm.get("objects", []):
            if o.get("name"):
                vlm_map[str(o["name"]).strip().lower()] = o

    props = {}
    for i, name in enumerate(names):
        v = vlm_map.get(name.strip().lower())
        if v and isinstance(v.get("density_kg_m3"), (int, float)):
            mat = str(v.get("material", "vlm"))
            dens = float(v["density_kg_m3"])
            fric = float(v.get("friction", DEFAULT_FRICTION))
            src = "vlm"
        else:
            mat, dens, fric = lookup(name)
            src = "lookup"
        dens = max(10.0, min(dens, 12000.0))
        fric = max(0.05, min(fric, 2.0))
        props[f"obj_{i}"] = {"name": name, "material": mat, "density": dens,
                             "friction": fric, "source": src}
        print(f"  obj_{i}: {name[:38]:40} {mat:10} density={dens:7.1f}  friction={fric:.2f}  [{src}]")

    out = a.output or (a.scene_dir / "object_properties.json")
    out.write_text(json.dumps(props, indent=2), encoding="utf-8")
    print(f"[props] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
