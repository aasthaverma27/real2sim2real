#!/usr/bin/env python3
"""Export a RoboSnap scene to MuJoCo MJCF.

Consumes the SF-Real2Sim refinement output, which stores each object as a
body-local mesh (``mesh_scaled.glb``) plus a rigid world pose
(``pose_optimized.txt``) and a convex decomposition (``collision/*.glb``).
That maps one-to-one onto MJCF: body pose + visual mesh geom + convex
collision geoms.

The support object (the table) becomes a static body; every other object
gets a free joint so it is dynamic.

    python scripts/export_mujoco.py --scene-dir outputs/robot_table_01 \
        --output-dir outputs/robot_table_01/mujoco
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh


def sanitize(name: str, fallback: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", (name or "").strip().lower()).strip("_")
    name = re.sub(r"^(the|a|an)_", "", name)
    return name[:40] or fallback


def pose_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 into position and a MuJoCo (w,x,y,z) quaternion."""
    pos = T[:3, 3].copy()
    R = T[:3, :3].copy()
    # re-orthonormalize: optimisation output can carry tiny non-rigid drift
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    M = np.eye(4)
    M[:3, :3] = R
    q = trimesh.transformations.quaternion_from_matrix(M)  # (w, x, y, z)
    return pos, q / np.linalg.norm(q)


def material_image(mesh: trimesh.Trimesh):
    """Texture image of a mesh. glTF gives PBRMaterial.baseColorTexture, not .image."""
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if mat is None:
        return None
    for attr in ("baseColorTexture", "image", "emissiveTexture"):
        img = getattr(mat, attr, None)
        if img is not None:
            return img
    return None


def mean_color(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    try:
        img = material_image(mesh)
        if img is not None:
            a = np.asarray(img.convert("RGB"), dtype=float) / 255.0
            return tuple(a.reshape(-1, 3).mean(axis=0))
        mat = getattr(mesh.visual, "material", None)
        for attr in ("baseColorFactor", "main_color"):
            c = getattr(mat, attr, None) if mat is not None else None
            if c is not None:
                c = np.asarray(c, dtype=float)
                if c.max() > 1.0:
                    c = c / 255.0
                return tuple(c[:3])
        vc = getattr(mesh.visual, "vertex_colors", None)
        if vc is not None and len(vc):
            return tuple(np.asarray(vc)[:, :3].mean(axis=0) / 255.0)
    except Exception:
        pass
    return (0.7, 0.7, 0.7)


def export_visual(mesh: trimesh.Trimesh, assets: Path, tag: str) -> tuple[str, bool]:
    """Export the visual mesh as OBJ. Returns (obj filename, textured).

    MuJoCo needs texture coordinates inside the mesh file, so the OBJ must be
    written with include_texture=True. trimesh names every texture
    ``material_0.png``, so each mesh is written to its own temp dir first and
    the PNG renamed per object.
    """
    obj_name = f"{tag}_visual.obj"
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(mesh.vertices)
    if has_uv and material_image(mesh) is not None:
        tmp = assets / f".tmp_{tag}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(tmp / "m.obj", include_texture=True)
            pngs = sorted(tmp.glob("*.png"))
            if pngs:
                shutil.move(str(tmp / "m.obj"), assets / obj_name)
                shutil.move(str(pngs[0]), assets / f"{tag}.png")
                return obj_name, True
        except Exception:
            pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    mesh.export(assets / obj_name, include_texture=False)
    return obj_name, False


def _world_zmin(mesh_path: Path, pose: np.ndarray) -> float:
    """Lowest world-space z of a body-local mesh under its pose."""
    m = trimesh.load(str(mesh_path), force="mesh")
    return float(trimesh.transform_points(np.asarray(m.vertices), pose)[:, 2].min())

def load_objects(scene_dir: Path) -> list[dict]:
    results = scene_dir / "refinement" / "sf_real2sim" / "results"
    if not results.is_dir():
        raise SystemExit(f"no refinement results in {results}")

    names: list[str] = []
    obj_txt = scene_dir / "object.txt"
    if obj_txt.is_file():
        names = [l.strip() for l in obj_txt.read_text(encoding="utf-8").splitlines() if l.strip()]

    # which object do others rest on -> that is the support (static) body
    support_idx = 0
    meta_path = scene_dir / "object_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            objs = meta.get("objects", meta) if isinstance(meta, dict) else meta
            parents = [o.get("support_parent_id", -1) for o in objs]
            referenced = [p for p in parents if p is not None and p >= 0]
            if referenced:
                support_idx = max(set(referenced), key=referenced.count)
        except Exception:
            pass

    out = []
    for d in sorted(results.iterdir(), key=lambda p: p.name):
        m = re.match(r"obj_(\d+)$", d.name)
        if not m or not (d / "mesh_scaled.glb").is_file():
            continue
        idx = int(m.group(1))
        pose_file = d / "pose_optimized.txt"
        if not pose_file.is_file():
            pose_file = d / "pose_initial_regularized.txt"
        out.append({
            "index": idx,
            "dir": d,
            "name": sanitize(names[idx] if idx < len(names) else "", f"object_{idx}"),
            "pose": np.loadtxt(pose_file),
            "hulls": sorted((d / "collision").glob("*.glb")),
            "is_support": idx == support_idx,
            "floor_z": _world_zmin(d / "mesh_scaled.glb", np.loadtxt(pose_file)),
        })
    return out


def build(scene_dir: Path, out_dir: Path, density: float, model_name: str,
          free_support: bool, static_names: list[str] | None = None,
          exclude_names: list[str] | None = None,
          materials: Path | None = None,
          include_robot: Path | None = None) -> Path:
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    static_names = static_names or []
    exclude_names = exclude_names or []
    props = {}
    if materials and Path(materials).is_file():
        props = json.loads(Path(materials).read_text(encoding="utf-8"))
        print(f"[mujoco] per-object properties from {materials}")
    objects = load_objects(scene_dir)
    if exclude_names:
        kept = [o for o in objects
                if not any(t in o["name"].lower() for t in exclude_names)]
        for o in objects:
            if o not in kept:
                print(f"[mujoco] excluding {o['name']}")
        objects = kept
    if not objects:
        raise SystemExit("no objects found")

    mujoco = ET.Element("mujoco", model=model_name)
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir="assets",
                  texturedir="assets", autolimits="true", balanceinertia="true")
    ET.SubElement(mujoco, "option", gravity="0 0 -9.81", integrator="implicitfast",
                  timestep="0.002")
    ET.SubElement(mujoco, "statistic", center="0 0 0.3", extent="1.5")
    vis = ET.SubElement(mujoco, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1920")
    ET.SubElement(vis, "quality", shadowsize="4096")

    default = ET.SubElement(mujoco, "default")
    dv = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(dv, "geom", type="mesh", contype="0", conaffinity="0",
                  group="2", density="0")
    dc = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(dc, "geom", type="mesh", group="3", condim="4",
                  friction="1.0 0.02 0.001", density=f"{density:g}", rgba="0.4 0.9 0.4 0.35")

    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.5 0.65 0.85", rgb2="0.1 0.15 0.25", width="512", height="512")
    ET.SubElement(asset, "texture", name="groundplane", type="2d", builtin="checker",
                  rgb1="0.25 0.26 0.27", rgb2="0.32 0.33 0.34",
                  width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texrepeat="4 4", reflectance="0.1")

    world = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true",
                  diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "light", pos="1 -1 2.5", dir="-0.4 0.4 -1", diffuse="0.3 0.3 0.3")
    # RoboSnap puts the support surface at z=0, so a plane at z=0 would coincide
    # with the tabletop. Drop the floor to the lowest point of the scene.
    floor_z = min(float(o["floor_z"]) for o in objects)
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  pos=f"0 0 {floor_z:.6f}", material="groundplane", condim="3")
    ET.SubElement(world, "camera", name="scene_view", pos="1.4 -1.4 1.2",
                  xyaxes="0.7 0.7 0 -0.35 0.35 0.87")
    if include_robot:
        # A reconstructed humanoid is a rigid mesh in a settled pose; the real
        # MJCF brings joints, limits and actuators.
        ET.SubElement(mujoco, "include", file=str(include_robot))
        print(f"[mujoco] including robot model: {include_robot}")

    summary = []
    for o in objects:
        tag = f"obj{o['index']}_{o['name']}"
        vis_mesh = trimesh.load(str(o["dir"] / "mesh_scaled.glb"), force="mesh")
        # Per-object metric correction. SAM3D scales each object independently,
        # so one global factor cannot fix a scene; measured references disagree
        # by 40-70% between objects.
        obj_scale = float(props.get(f"obj_{o['index']}", {}).get("scale", 1.0))
        if obj_scale != 1.0:
            vis_mesh.apply_scale(obj_scale)

        vis_file, textured = export_visual(vis_mesh, assets, tag)
        ET.SubElement(asset, "mesh", name=f"{tag}_visual", file=vis_file)

        tex_name = None
        if textured:
            tex_name = f"{tag}_tex"
            ET.SubElement(asset, "texture", name=tex_name, type="2d", file=f"{tag}.png")
            ET.SubElement(asset, "material", name=f"{tag}_mat", texture=tex_name,
                          texuniform="false", specular="0.1", shininess="0.1")

        hull_names = []
        for k, h in enumerate(o["hulls"]):
            hm = trimesh.load(str(h), force="mesh")
            if hm.volume <= 0 or len(hm.vertices) < 4:
                continue
            if obj_scale != 1.0:
                hm.apply_scale(obj_scale)
            hf = f"{tag}_coll_{k}.obj"
            hm.export(assets / hf, include_texture=False)
            hn = f"{tag}_coll_{k}"
            ET.SubElement(asset, "mesh", name=hn, file=hf)
            hull_names.append(hn)

        pos, quat = pose_to_pos_quat(o["pose"])
        body = ET.SubElement(world, "body", name=tag,
                             pos=" ".join(f"{v:.6f}" for v in pos),
                             quat=" ".join(f"{v:.6f}" for v in quat))
        forced_static = any(t in o["name"].lower() for t in static_names)
        dynamic = ((not o["is_support"]) or free_support) and not forced_static
        if dynamic:
            ET.SubElement(body, "freejoint", name=f"{tag}_free")

        gv = {"class": "visual", "mesh": f"{tag}_visual"}
        if tex_name:
            gv["material"] = f"{tag}_mat"
        else:
            r, g, b = mean_color(vis_mesh)
            gv["rgba"] = f"{r:.3f} {g:.3f} {b:.3f} 1"
        ET.SubElement(body, "geom", name=f"{tag}_vis", **gv)

        pr = props.get(f"obj_{o['index']}", {})
        coll_attrs = {"class": "collision"}
        if pr:
            coll_attrs["density"] = f"{float(pr.get('density', density)):g}"
            fr = float(pr.get("friction", 1.0))
            coll_attrs["friction"] = f"{fr:g} 0.02 0.001"
        for hn in hull_names:
            ET.SubElement(body, "geom", dict(coll_attrs, mesh=hn, name=hn + "_g"))
        if not hull_names:  # fall back to the convex hull of the visual mesh
            ET.SubElement(body, "geom", dict(coll_attrs, mesh=f"{tag}_visual",
                                             name=f"{tag}_coll_fallback"))
        summary.append((tag, "static" if not dynamic else "free", len(hull_names),
                        pr.get("density"), pr.get("material")))

    xml_path = out_dir / "scene.xml"
    ET.indent(mujoco, space="  ")
    xml_path.write_bytes(ET.tostring(mujoco, encoding="utf-8", xml_declaration=True))

    print(f"[mujoco] wrote {xml_path}")
    print(f"[mujoco] assets: {len(list(assets.iterdir()))} files in {assets}")
    for tag, kind, nh, dens, mat in summary:
        extra = f"  {mat} density={dens:g}" if dens else ""
        print(f"  {tag:34} {kind:6} collision_hulls={nh}{extra}")
    return xml_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--density", type=float, default=300.0,
                   help="kg/m^3 applied to collision geoms (default: 300, light plastic)")
    p.add_argument("--model-name", default="robosnap_scene")
    p.add_argument("--free-support", action="store_true",
                   help="Give the support object a free joint too (default: static).")
    p.add_argument("--static", default="", metavar="SUBSTR,...",
                   help="Comma-separated name substrings to make static (no free joint). "
                        "Useful for a reconstructed robot, which would otherwise topple.")
    p.add_argument("--materials", type=Path, metavar="JSON",
                   help="object_properties.json from scripts/estimate_properties.py: "
                        "per-object density and friction instead of one global value.")
    p.add_argument("--include-robot", type=Path, metavar="MJCF",
                   help="Path to a robot MJCF to <include> in the scene, e.g. the "
                        "official Unitree model with real joints and actuators.")
    p.add_argument("--exclude", default="", metavar="SUBSTR,...",
                   help="Comma-separated name substrings to omit entirely, e.g. 'robot' "
                        "when you intend to load the real URDF/MJCF instead.")
    a = p.parse_args()
    out = a.output_dir or (a.scene_dir / "mujoco")
    split = lambda v: [t.strip().lower() for t in v.split(",") if t.strip()]
    build(a.scene_dir, out, a.density, a.model_name, a.free_support,
          split(a.static), split(a.exclude), a.materials, a.include_robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
