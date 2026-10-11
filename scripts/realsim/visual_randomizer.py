"""Visual domain randomization for the photoreal renderers.

Lights, material colour/gloss, robot greys, an optional table texture swap and
a colour jitter on the splat background - everything that changes what the
cameras see and nothing that changes the physics. That last part is the point:
the demos are replayed through the WBC, so anything that moved an object,
resized it or changed its friction would break the replay it is decorating.

Who does what:

  * run_mujoco_wbc_sim.py owns the episode. On startup and on every reset
    (backspace, or a reset request) it picks a new seed and broadcasts it on the
    object-stream ports as a tiny RSVR datagram. It never renders, so it needs
    none of this file except the wire format.
  * every renderer (vla_camera_server, mugs_viewer, elp_camera_publisher) sees
    the seed through its MirrorReceiver and calls VisualDRFollower.update(). The
    parameters are a pure function of (seed, config), so all windows and the
    recorded cameras agree without talking to each other, and any episode can
    be re-derived later from its seed alone.

Seed 0 is the nominal scene, which is what the sim sends before the first reset.

Ranges live in configs/visual_randomization.yaml. `sample()` is deterministic
per seed; `apply()` writes the model; `upload()` pushes swapped textures to the
GL contexts. None of it touches qpos, geometry, masses or contacts.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

VIS_MAGIC = b"RSVR"
# magic + uint32 seed + uint32 episode counter. 12 bytes: never mistaken for a
# 51-float state packet (204 bytes) or an RSOB object packet, so older
# consumers (collection.py, anything built on MirrorReceiver) drop it unread.
VIS_FMT = "<II"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "visual_randomization.yaml"


def pack_visual(seed: int, episode: int) -> bytes:
    return VIS_MAGIC + struct.pack(VIS_FMT, int(seed) & 0xFFFFFFFF,
                                   int(episode) & 0xFFFFFFFF)


def unpack_visual(pkt: bytes):
    """-> (seed, episode) or None if the datagram is not an RSVR one."""
    if len(pkt) != 4 + struct.calcsize(VIS_FMT) or pkt[:4] != VIS_MAGIC:
        return None
    return struct.unpack(VIS_FMT, pkt[4:])


def load_config(path=None) -> dict:
    import yaml
    p = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _u(rng, rr):
    """Uniform draw from a [lo, hi] pair, or the value itself if scalar."""
    if isinstance(rr, (list, tuple)):
        return float(rng.uniform(float(rr[0]), float(rr[1])))
    return float(rr)


def jitter_image(rgb: np.ndarray, bg: dict) -> np.ndarray:
    """Brightness / contrast / saturation / per-channel gain on a uint8 image."""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    x = (x - 0.5) * bg["contrast"] + 0.5
    x *= bg["brightness"]
    grey = x.mean(axis=-1, keepdims=True)
    x = grey + (x - grey) * bg["saturation"]
    x *= np.asarray(bg["gain"], np.float32)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class VisualRandomizer:
    """Sample and apply one episode's look to a render-scene MjModel."""

    def __init__(self, model, cfg: dict):
        import mujoco
        self.cfg = cfg
        name = lambda t, i: mujoco.mj_id2name(model, t, i) or ""

        # --- nominal values, so seed 0 / nominal episodes restore exactly ---
        self.nom = {
            "light_pos": model.light_pos.copy(),
            "light_dir": model.light_dir.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "light_ambient": model.light_ambient.copy(),
            "light_specular": model.light_specular.copy(),
            "light_castshadow": model.light_castshadow.copy(),
            "light_cutoff": model.light_cutoff.copy(),
            "head_ambient": np.array(model.vis.headlight.ambient),
            "head_diffuse": np.array(model.vis.headlight.diffuse),
            "mat_rgba": model.mat_rgba.copy(),
            "mat_specular": model.mat_specular.copy(),
            "mat_shininess": model.mat_shininess.copy(),
            "geom_rgba": model.geom_rgba.copy(),
        }

        # --- lights: the render scene has one directional (key) and one spot ---
        ltype = np.asarray(model.light_type).astype(int)
        dirs = [i for i in range(model.nlight)
                if ltype[i] == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)]
        spots = [i for i in range(model.nlight)
                 if ltype[i] == int(mujoco.mjtLightType.mjLIGHT_SPOT)]
        self.key = dirs[0] if dirs else None
        self.spot = spots[0] if spots else None
        self.extra = [i for i in range(model.nlight) if i not in (self.key, self.spot)]

        # --- aim point: top centre of the table, measured, no frame constants ---
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.aim = self._table_top(model, data, cfg.get("aim_body", "table"))

        # --- materials, by name substring ---
        self.mats = {}
        for key, mc in (cfg.get("materials") or {}).items():
            ids = [i for i in range(model.nmat)
                   if mc.get("match", key) in name(mujoco.mjtObj.mjOBJ_MATERIAL, i)]
            if ids:
                self.mats[key] = ids

        # --- robot greys: geoms with no material, bucketed by nominal grey ---
        self.greys = []
        for gc in cfg.get("robot_grey") or []:
            g = float(gc["nominal"])
            ids = [i for i in range(model.ngeom)
                   if int(model.geom_matid[i]) < 0
                   and int(model.geom_group[i]) != 3
                   and np.allclose(model.geom_rgba[i, :3], g, atol=0.011)]
            if ids:
                self.greys.append((gc, np.array(ids)))

        # --- optional texture swaps: {key: (texid, [image paths])} ---
        self.tex = {}
        self.tex_nominal = {}
        self.tex_current = {}
        for key, tc in (cfg.get("textures") or {}).items():
            pool = Path(tc.get("pool", ""))
            if not pool.is_absolute():
                pool = ROOT / pool
            files = sorted(f for f in pool.glob("*")
                           if f.suffix.lower() in (".png", ".jpg", ".jpeg")) \
                if pool.is_dir() else []
            tid = next((i for i in range(model.ntex)
                        if tc.get("match", key) in name(mujoco.mjtObj.mjOBJ_TEXTURE, i)
                        and int(model.tex_type[i]) == int(mujoco.mjtTexture.mjTEXTURE_2D)),
                       -1)
            if tid < 0 or not files:
                print(f"[visual-dr] texture '{key}': "
                      f"{'no matching texture' if tid < 0 else f'no images in {pool}'}"
                      f" - swap disabled", flush=True)
                continue
            adr = int(model.tex_adr[tid])
            n = int(model.tex_width[tid] * model.tex_height[tid] * model.tex_nchannel[tid])
            self.tex[key] = (tid, files, float(tc.get("prob", 0.5)))
            self.tex_nominal[key] = model.tex_data[adr:adr + n].copy()
            self.tex_current[key] = None

        print(f"[visual-dr] key light {self.key}, spot {self.spot}, "
              f"aim {np.round(self.aim, 3).tolist()}, "
              f"materials {sorted(self.mats)}, "
              f"grey buckets {[len(ids) for _, ids in self.greys]}, "
              f"texture pools {({k: len(v[1]) for k, v in self.tex.items()})}",
              flush=True)

    @staticmethod
    def _table_top(model, data, needle):
        import mujoco
        for b in range(model.nbody):
            if needle not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""):
                continue
            top = -np.inf
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) != b:
                    continue
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(model.geom_dataid[g])
                    v0, n0 = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                    V = (model.mesh_vert[v0:v0 + n0].reshape(-1, 3)
                         @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
                    top = max(top, float(V[:, 2].max()))
            xy = data.xpos[b][:2]
            return np.array([xy[0], xy[1], top if np.isfinite(top) else data.xpos[b][2]])
        return np.array(model.stat.center, float)

    # ------------------------------------------------------------------
    def sample(self, seed: int) -> dict:
        """The parameters for one episode. Pure function of (seed, config)."""
        c = self.cfg
        rng = np.random.default_rng(int(seed))
        if seed == 0 or rng.uniform() < float(c.get("nominal_prob", 0.0)):
            return {"seed": int(seed), "nominal": True}

        guard = c.get("brightness_guard", [0.0, 1e9])
        for _ in range(50):
            p = {"seed": int(seed), "nominal": False}
            tint = lambda: [_u(rng, c.get("color_tint", 1.0)) for _ in range(3)]

            if self.key is not None:
                k = c["key_light"]
                el = np.radians(_u(rng, k["elevation_deg"]))
                az = np.radians(_u(rng, k["azimuth_deg"]))
                d = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                p["key"] = {"dir": d.tolist(), "diffuse": _u(rng, k["diffuse"]),
                            "ambient": _u(rng, k.get("ambient", 0.0)),
                            "specular": _u(rng, k.get("specular", 0.3)),
                            "tint": tint()}
            if self.spot is not None:
                s = c["spot_light"]
                r = float(s.get("offset_xy_m", 0.0))
                off = rng.uniform(-r, r, 2)
                pos = np.array([self.aim[0] + off[0], self.aim[1] + off[1],
                                self.aim[2] + _u(rng, s["height_m"])])
                d = self.aim - pos
                p["spot"] = {"pos": pos.tolist(), "dir": (d / np.linalg.norm(d)).tolist(),
                             "diffuse": _u(rng, s["diffuse"]),
                             "ambient": _u(rng, s.get("ambient", 0.0)),
                             "specular": _u(rng, s.get("specular", 0.3)),
                             "cutoff": _u(rng, s.get("cutoff_deg", 45.0)),
                             "castshadow": bool(rng.uniform() < float(s.get("castshadow_prob", 1.0))),
                             "tint": tint()}
            h = c.get("headlight") or {}
            p["head"] = {"ambient": _u(rng, h.get("ambient", 0.1)),
                         "diffuse": _u(rng, h.get("diffuse", 0.4))}

            total = (p.get("key", {}).get("diffuse", 0.0)
                     + p.get("spot", {}).get("diffuse", 0.0) + p["head"]["diffuse"])
            if float(guard[0]) <= total <= float(guard[1]):
                break
        p["total_diffuse"] = round(total, 3)

        p["materials"] = {}
        for key, mc in (c.get("materials") or {}).items():
            if key not in self.mats:
                continue
            b = _u(rng, mc.get("brightness", 1.0))
            j = float(mc.get("channel_jitter", 0.0))
            p["materials"][key] = {
                "rgb": [b * float(rng.uniform(1 - j, 1 + j)) for _ in range(3)],
                "specular": _u(rng, mc.get("specular", 0.1)),
                "shininess": _u(rng, mc.get("shininess", 0.1)),
            }
        p["grey"] = [_u(rng, gc["range"]) for gc, _ in self.greys]

        p["textures"] = {}
        for key, (_, files, prob) in self.tex.items():
            if rng.uniform() < prob:
                p["textures"][key] = str(files[int(rng.integers(len(files)))])

        bg = c.get("background") or {}
        p["background"] = {"brightness": _u(rng, bg.get("brightness", 1.0)),
                           "contrast": _u(rng, bg.get("contrast", 1.0)),
                           "saturation": _u(rng, bg.get("saturation", 1.0)),
                           "gain": [_u(rng, bg.get("channel_gain", 1.0)) for _ in range(3)]}
        return p

    # ------------------------------------------------------------------
    def apply(self, model, p: dict) -> list:
        """Write one episode's look into the model. Returns texture ids to upload."""
        n = self.nom
        model.light_pos[:] = n["light_pos"]
        model.light_dir[:] = n["light_dir"]
        model.light_diffuse[:] = n["light_diffuse"]
        model.light_ambient[:] = n["light_ambient"]
        model.light_specular[:] = n["light_specular"]
        model.light_castshadow[:] = n["light_castshadow"]
        model.light_cutoff[:] = n["light_cutoff"]
        model.vis.headlight.ambient[:] = n["head_ambient"]
        model.vis.headlight.diffuse[:] = n["head_diffuse"]
        model.mat_rgba[:] = n["mat_rgba"]
        model.mat_specular[:] = n["mat_specular"]
        model.mat_shininess[:] = n["mat_shininess"]
        model.geom_rgba[:] = n["geom_rgba"]

        want_tex = {} if p.get("nominal") else p.get("textures", {})
        dirty = self._apply_textures(model, want_tex)
        if p.get("nominal"):
            return dirty

        if "key" in p and self.key is not None:
            k, i = p["key"], self.key
            d = np.asarray(k["dir"])
            model.light_dir[i] = d
            # A directional light's shadow map is centred on its position, so
            # keep it upstream of the table along the new direction.
            model.light_pos[i] = self.aim - 3.0 * d
            model.light_diffuse[i] = k["diffuse"] * np.asarray(k["tint"])
            model.light_ambient[i] = k["ambient"]
            model.light_specular[i] = k["specular"]
            model.light_castshadow[i] = 1
        if "spot" in p and self.spot is not None:
            s, i = p["spot"], self.spot
            model.light_pos[i] = s["pos"]
            model.light_dir[i] = s["dir"]
            model.light_diffuse[i] = s["diffuse"] * np.asarray(s["tint"])
            model.light_ambient[i] = s["ambient"]
            model.light_specular[i] = s["specular"]
            model.light_cutoff[i] = s["cutoff"]
            model.light_castshadow[i] = int(s["castshadow"])
        h = p["head"]
        model.vis.headlight.ambient[:] = h["ambient"]
        model.vis.headlight.diffuse[:] = h["diffuse"]

        for key, mp in p.get("materials", {}).items():
            for m in self.mats.get(key, []):
                model.mat_rgba[m, :3] = np.clip(n["mat_rgba"][m, :3] * mp["rgb"], 0, 1)
                model.mat_specular[m] = mp["specular"]
                model.mat_shininess[m] = mp["shininess"]
        for (gc, ids), g in zip(self.greys, p.get("grey", [])):
            model.geom_rgba[ids, :3] = g
        return dirty

    def _apply_textures(self, model, want: dict) -> list:
        dirty = []
        for key, (tid, _, _) in self.tex.items():
            path = want.get(key)
            if path == self.tex_current[key]:
                continue
            adr = int(model.tex_adr[tid])
            w, h, c = (int(model.tex_width[tid]), int(model.tex_height[tid]),
                       int(model.tex_nchannel[tid]))
            if path is None:
                buf = self.tex_nominal[key]
            else:
                import cv2
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[visual-dr] cannot read {path}; keeping the current texture",
                          flush=True)
                    continue
                img = cv2.resize(img[:, :, ::-1], (w, h), interpolation=cv2.INTER_AREA)
                if c == 4:
                    img = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], -1)
                buf = np.ascontiguousarray(img[:, :, :c]).reshape(-1)
            model.tex_data[adr:adr + w * h * c] = buf
            self.tex_current[key] = path
            dirty.append(tid)
        return dirty


INFO_MAGIC = b"RSVI"        # + JSON {seed, episode, summary}: a look, for humans


def summarize(p: dict) -> str:
    """One look, as the lines an operator reads before save/discard."""
    if p.get("nominal"):
        return "nominal scene (original lighting and colours)"
    out = []
    if "key" in p:
        k = p["key"]
        d = np.asarray(k["dir"])
        el = np.degrees(np.arcsin(np.clip(-d[2], -1, 1)))
        az = np.degrees(np.arctan2(-d[1], -d[0])) % 360
        t = k["tint"]
        warm = "warm" if t[0] > t[2] + 0.05 else "cool" if t[2] > t[0] + 0.05 else "neutral"
        out.append(f"key light  : diffuse {k['diffuse']:.2f}, elevation {el:.0f} deg "
                   f"(90 = overhead), azimuth {az:.0f} deg, {warm} tint "
                   f"rgb({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})")
    if "spot" in p:
        s = p["spot"]
        out.append(f"spot light : diffuse {s['diffuse']:.2f}, cone {s['cutoff']:.0f} deg, "
                   f"at z {s['pos'][2]:.2f} m, shadows {'on' if s['castshadow'] else 'off'}")
    h = p["head"]
    out.append(f"headlight  : diffuse {h['diffuse']:.2f}, ambient {h['ambient']:.2f}"
               f"   | total light {p.get('total_diffuse', 0):.2f} (nominal 1.40)")
    mats = p.get("materials", {})
    if mats:
        out.append("materials  : " + ", ".join(
            f"{k} x{np.mean(v['rgb']):.2f} gloss {v['specular']:.2f}" for k, v in mats.items()))
    if p.get("grey"):
        out.append("robot greys: " + ", ".join(f"{g:.2f}" for g in p["grey"]))
    bg = p.get("background")
    if bg:
        out.append(f"background : brightness {bg['brightness']:.2f}, contrast "
                   f"{bg['contrast']:.2f}, saturation {bg['saturation']:.2f}")
    if p.get("textures"):
        out.append("textures   : " + ", ".join(f"{k}={Path(v).name}"
                                               for k, v in p["textures"].items()))
    return "\n".join(out)


class VisualDRFollower:
    """Glue for a renderer: watch the sim's seed, re-skin the model on change.

        follower = VisualDRFollower(model, "configs/visual_randomization.yaml",
                                    [sensor, ego_sensor], log="...jsonl")
        ...
        mirror.apply(data)
        follower.update(mirror)
        img = sensor.render(model, data, cam)
    """

    def __init__(self, model, config=None, sensors=(), log=None, tag="render",
                 report_ports=()):
        self.model = model
        self.rand = VisualRandomizer(model, load_config(config))
        self.sensors = []
        # Where to send each look's human-readable summary (collection.py's
        # object-stream port). Re-sent every second: UDP, and a listener may
        # start after the look was applied.
        self.report = [("127.0.0.1", int(p)) for p in report_ports if p]
        self._report_sock = None
        self._report_next = 0.0
        if self.report:
            import socket
            self._report_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.current = None           # (seed, episode) last applied
        self.params = {"seed": 0, "nominal": True}
        self.log = Path(log) if log else None
        self.tag = tag
        for s in sensors:
            self.attach(s)

    def attach(self, sensor):
        """Route a GaussianSensor's splat background through the jitter.

        Wraps the bound method on the instance, so MuGS itself is untouched,
        and the static-background cache keeps working: it is simply dropped
        whenever the look changes.
        """
        if sensor is None or any(s is sensor for s in self.sensors):
            return
        orig = sensor._render_3dgs_background

        def wrapped(camera_params, _orig=orig):
            rgb, depth = _orig(camera_params)
            bg = self.params.get("background")
            if bg is not None and not self.params.get("nominal"):
                rgb = jitter_image(rgb, bg)
            return rgb, depth

        sensor._render_3dgs_background = wrapped
        self.sensors.append(sensor)

    def send_report(self, force=False):
        if self._report_sock is None or self.current is None:
            return
        now = time.time()
        if not force and now < self._report_next:
            return
        self._report_next = now + 1.0
        seed, episode = self.current
        pkt = INFO_MAGIC + json.dumps({"seed": int(seed), "episode": int(episode),
                                       "summary": summarize(self.params)}).encode()
        for d in self.report:
            try:
                self._report_sock.sendto(pkt, d)
            except OSError:
                pass

    def update(self, mirror) -> bool:
        v = getattr(mirror, "visual", None)
        if v is None or v == self.current:
            self.send_report()
            return False
        seed, episode = v
        self.params = self.rand.sample(seed)
        dirty = self.rand.apply(self.model, self.params)
        self.upload(dirty)
        for s in self.sensors:
            s._cached_background = None
        self.current = v
        desc = ("nominal" if self.params["nominal"] else
                f"key {self.params.get('key', {}).get('diffuse', 0):.2f} "
                f"spot {self.params.get('spot', {}).get('diffuse', 0):.2f} "
                f"head {self.params['head']['diffuse']:.2f}"
                f"{' tex ' + ','.join(self.params['textures']) if self.params.get('textures') else ''}")
        print(f"[visual-dr] episode {episode} seed {seed}: {desc}", flush=True)
        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.time(), "tag": self.tag,
                                    "episode": episode, **self.params}) + "\n")
        self.send_report(force=True)
        return True

    def upload(self, texids):
        import mujoco
        if not texids:
            return
        for s in self.sensors:
            r = getattr(s, "_renderer", None)
            if r is None:        # created later: it uploads model.tex_data itself
                continue
            r._gl_context.make_current()
            for t in texids:
                mujoco.mjr_uploadTexture(self.model, r._mjr_context, int(t))
