"""Reset-time layout randomizer for the plate + fruits + basket scenes.

TableRandomizer (randomize_objects.py) places an apple and a basket anywhere in
an annulus around the robot. This scene has a different shape of variety: two
fixed spots on the table, taken from the photo the scene was built from, which
the plate and the basket either keep or swap (mirrored across the robot's centre
line, so the basket stays as near the robot either way); a small jitter around each spot;
and the fruits shuffled on the plate. Everything stays near where it was
photographed, so the layouts never wander far from the real one.

Configured by <scene>/layout_random.json (shipped with the scene, e.g.
scenes/new_table_env/layout_random.json):

    {"table": "obj0_white_table", "plate": "obj3_plate",
     "basket": "obj2_brown_basket", "fruits": ["obj4_mango", ...],
     "swap_prob": 0.5, "swap_mode": "mirror", "jitter_along_m": 0.04, "jitter_depth_m": 0.04,
     "basket_yaw_deg": 15, "plate_rim_m": 0.015, "min_gap_m": 0.01,
     "robot_clearance_m": 0.01}

The two spots are the plate's and the basket's positions in the scene file, read
in the table's frame, so they follow the table wherever it is placed. Same
interface as TableRandomizer where run_mujoco_wbc_sim.py uses it: `apply(data)`
returns {name: xy}, `zero_velocities(model, data)`, `items`, `top_z`.
"""

from __future__ import annotations

import numpy as np


def _yaw_q(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])


class PlateBasketRandomizer:
    def __init__(self, model, data, cfg: dict, seed=None):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.rng = np.random.default_rng(seed)
        body = lambda b: model.body(b).name

        def find(needle):
            for b in range(model.nbody):
                if needle in body(b):
                    return b
            raise ValueError(f"no body matching {needle!r}")

        # The robot's floating base: "floating_base_joint" in the sim scene,
        # "g1_floating_base_joint" in the render scene.
        fb = next((j for j in range(model.njnt)
                   if model.joint(j).name.endswith("floating_base_joint")), None)
        if fb is None:
            raise ValueError("no *floating_base_joint in the model")
        self.fb = fb

        tb = find(cfg["table"])
        self.t_xy = data.xpos[tb][:2].copy()
        R = data.xmat[tb].reshape(3, 3)
        self.ax = R[:2, 0] / np.linalg.norm(R[:2, 0])          # along the table
        self.ay = np.array([-self.ax[1], self.ax[0]])           # across it
        # table-top half extents, from its collision geoms in the table frame
        pts = self._geom_points(data, tb)
        rel = pts[:, :2] - self.t_xy
        self.half = np.array([np.abs(rel @ self.ax).max(), np.abs(rel @ self.ay).max()])
        self.top_z = float(pts[:, 2].max())

        # Fruits left out of this run (run_mujoco_wbc_sim.py --fruits): parked
        # on the floor 4 m to the side of the table, where nothing can reach
        # them; the renderers are told not to draw them (start.sh). Ordinary
        # physics - body_gravcomp cannot be switched on after compile.
        # ["none"] parks every fruit: an empty plate (start.sh's default).
        want = cfg.get("only_fruits")
        empty = want == ["none"]
        hidden = [f for f in cfg["fruits"]
                  if empty or (want and not any(w in f for w in want))]
        cfg = dict(cfg, fruits=[f for f in cfg["fruits"] if f not in hidden])
        if not cfg["fruits"] and not empty:
            raise ValueError(f"no fruit matches {want}; have {hidden}")
        self.cfg = cfg
        self.parked = []
        for needle in hidden:
            b = find(needle)
            j = int(model.body_jntadr[b])
            adr = int(model.jnt_qposadr[j])
            # resting height above the floor = its height above the plate's table
            z = float(data.qpos[adr + 2] - self.top_z + 0.01)
            self.parked.append((adr, int(model.jnt_dofadr[j]), body(b), z))

        self.items = []
        for key in ["plate", "basket"] + list(cfg["fruits"]):
            needle = cfg[key] if key in ("plate", "basket") else key
            b = find(needle)
            j = int(model.body_jntadr[b])
            adr = int(model.jnt_qposadr[j])
            q = data.qpos[adr:adr + 7].copy()
            p = self._geom_points(data, b)
            foot = float(np.linalg.norm(p[:, :2] - data.xpos[b][:2], axis=1).max())
            self.items.append({"name": body(b), "key": key, "body": b, "adr": adr,
                               "dof": int(model.jnt_dofadr[j]), "q0": q,
                               "quat": q[3:7].copy(), "foot": foot,
                               "lift": float(q[2] - self.top_z),
                               "local": p - data.xpos[b]})
        self.it = {i["key"]: i for i in self.items}
        self.last = None
        # the two spots, table frame
        self.spot = {k: self._to_table(self.it[k]["q0"][:2]) for k in ("plate", "basket")}
        # Swapped layout: by default ("swap_mode": "mirror") each object goes to
        # its own spot mirrored across the robot's centre line, so the basket is
        # as near the robot on the left as it is on the right. Exchanging the two
        # spots instead ("exchange") throws the basket out to where the plate
        # was. The mirrored plate is pulled in to stay on the table.
        fb = self.fb
        rx = self._to_table(model.qpos0[model.jnt_qposadr[fb]:model.jnt_qposadr[fb] + 2])[0]
        r_p = self.it["plate"]["foot"]
        if cfg.get("swap_mode", "mirror") == "mirror":
            px = np.clip(2 * rx - self.spot["plate"][0], -(self.half[0] - r_p - 0.01),
                         self.half[0] - r_p - 0.01)
            self.spot_swap = {"plate": np.array([px, self.spot["plate"][1]]),
                              "basket": np.array([2 * rx - self.spot["basket"][0],
                                                  self.spot["basket"][1]])}
        else:
            self.spot_swap = {"plate": self.spot["basket"], "basket": self.spot["plate"]}
        bl = self.it["basket"]["local"][:, :2] @ data.xmat[self.it["basket"]["body"]].reshape(3, 3)[:2, :2]
        self.basket_half = np.abs(bl).max(0)                   # body frame x, y
        self.fruit_off = {k: self.it[k]["q0"][:2] - self.it["plate"]["q0"][:2]
                          for k in cfg["fruits"]}
        # Objects the scene built beside the plate rather than on it (too big for
        # the dish, e.g. the unicorn): never shuffled onto the plate - each reset
        # puts them back where they were built.
        self.fixed = [k for k in cfg["fruits"]
                      if np.linalg.norm(self.fruit_off[k]) > self.it["plate"]["foot"]]
        self.cfg = cfg = dict(cfg, fruits=[k for k in cfg["fruits"] if k not in self.fixed])
        # Off-plate objects with a region in the config ("side_items", robot frame,
        # e.g. the unicorn on the right-hand side) get a fresh spot every reset,
        # clear of the hands, the plate, the basket and whatever else is out.
        fb = self.fb
        qa = model.jnt_qposadr[fb]; w, x, y, z = data.qpos[qa + 3:qa + 7]
        f = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)]); self.r_fwd = f / np.linalg.norm(f)
        self.r_left = np.array([-self.r_fwd[1], self.r_fwd[0]]); self.r_xy = data.qpos[qa:qa + 2].copy()
        self.side = {k: r for k in self.fixed for n, r in cfg.get("side_items", {}).items() if n in k}
        robot_root = model.body_rootid[model.jnt_bodyid[fb]]
        self.robot = [g for g in range(model.ngeom) if model.geom_contype[g]
                      and model.body_rootid[model.geom_bodyid[g]] == robot_root]
        self.objs = {i["key"]: [g for g in range(model.ngeom)
                                if model.geom_bodyid[g] == i["body"] and model.geom_contype[g]]
                     for i in self.items}
        print(f"[layout] plate/basket spots (table frame) plate {np.round(self.spot['plate'], 3)} "
              f"basket {np.round(self.spot['basket'], 3)}; swapped plate "
              f"{np.round(self.spot_swap['plate'], 3)} basket {np.round(self.spot_swap['basket'], 3)} "
              f"({cfg.get('swap_mode', 'mirror')}); swap p={cfg['swap_prob']}, "
              f"jitter +-{cfg['jitter_along_m']*100:.0f} cm along, "
              f"+-{cfg['jitter_depth_m']*100:.0f} cm depth, basket yaw "
              f"+-{cfg['basket_yaw_deg']} deg, fruits {[i['name'] for i in self.items[2:]]} "
              f"shuffled on the plate" + (f"; hidden {[p[2] for p in self.parked]}"
                                          if self.parked else ""))

    # ------------------------------------------------------------------
    def _geom_points(self, data, b):
        m = self.m
        out = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != self.mj.mjtGeom.mjGEOM_MESH:
                continue
            mi = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            out.append(V @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        return np.vstack(out)

    def _to_table(self, xy):
        v = np.asarray(xy) - self.t_xy
        return np.array([v @ self.ax, v @ self.ay])

    def _to_world(self, t):
        return self.t_xy + self.ax * t[0] + self.ay * t[1]

    def _basket_corners(self, c, yaw):
        hx, hy = self.basket_half
        k = np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]])
        cs, sn = np.cos(yaw), np.sin(yaw)
        return c + k @ np.array([[cs, sn], [-sn, cs]])

    # ------------------------------------------------------------------
    def _sample(self, rng, swap):
        c = self.cfg
        jit = lambda: np.array([rng.uniform(-1, 1) * c["jitter_along_m"],
                                rng.uniform(-1, 1) * c["jitter_depth_m"]])
        spot = self.spot_swap if swap else self.spot
        plate_t = spot["plate"] + jit()
        basket_t = spot["basket"] + jit()
        byaw = np.radians(rng.uniform(-1, 1) * c["basket_yaw_deg"])
        m = 0.01
        r_p = self.it["plate"]["foot"]
        if np.any(np.abs(plate_t) + r_p > self.half - m):
            return None
        bc = self._basket_corners(basket_t, byaw)            # basket yaw in table frame
        if np.any(np.abs(bc) > self.half - m):
            return None
        # plate circle vs basket rectangle
        cs, sn = np.cos(byaw), np.sin(byaw)
        local = np.array([[cs, sn], [-sn, cs]]) @ (plate_t - basket_t)
        nearest = np.clip(local, -self.basket_half, self.basket_half)
        if np.linalg.norm(local - nearest) < r_p + c["min_gap_m"]:
            return None
        # fruits: random, non-overlapping, inside the rim
        r_in = r_p - c["plate_rim_m"]
        fr = {}
        for k in sorted(self.cfg["fruits"], key=lambda k: -self.it[k]["foot"]):
            f = self.it[k]["foot"] * 0.8          # footprint is the circumscribed radius
            for _ in range(200):
                rr = (r_in - f) * np.sqrt(rng.uniform())
                a = rng.uniform(0, 2 * np.pi)
                p = rr * np.array([np.cos(a), np.sin(a)])
                if all(np.linalg.norm(p - q) >= f + self.it[o]["foot"] * 0.8 + 0.004
                       for o, q in fr.items()):
                    fr[k] = p
                    break
            else:
                return None
        return swap, plate_t, basket_t, byaw, fr

    def _write(self, data, s, rng):
        swap, plate_t, basket_t, byaw, fr = s
        tyaw = np.arctan2(self.ax[1], self.ax[0])
        pw = self._to_world(plate_t)
        put = {"plate": (pw, rng.uniform(0, 2 * np.pi), True),
               "basket": (self._to_world(basket_t), byaw, False)}
        for k, p in fr.items():
            # plate-local offsets are table-frame; rotate into the world
            put[k] = (pw + self.ax * p[0] + self.ay * p[1], rng.uniform(0, 2 * np.pi), True)
        for k, (xy, yaw, absolute) in put.items():
            it = self.it[k]
            q0 = it["q0"]
            data.qpos[it["adr"]:it["adr"] + 2] = xy
            data.qpos[it["adr"] + 2] = q0[2]
            # basket: a small turn off its built pose; plate, fruits: any heading
            quat = _qmul(_yaw_q(yaw), q0[3:7])
            data.qpos[it["adr"] + 3:it["adr"] + 7] = quat / np.linalg.norm(quat)
        return swap

    def _robot_gap(self, data):
        mj, fr = self.mj, np.zeros(6)
        return min(mj.mj_geomDistance(self.m, data, r, g, 0.1, fr)
                   for r in self.robot for k in ("plate", "basket") for g in self.objs[k])

    def park(self, data):
        """Send the left-out fruits out of the way (again, after a reset)."""
        for i, (adr, dof, _, z) in enumerate(self.parked):
            xy = self.t_xy + self.ax * (4.0 + 0.3 * i)
            data.qpos[adr:adr + 3] = [xy[0], xy[1], z]
            data.qvel[dof:dof + 6] = 0.0

    def _place_side(self, data, rng):
        """Random spot for each side item; built spot if nothing fits."""
        mj, fr = self.mj, np.zeros(6)
        out = {it["key"] for it in self.items} - {p[2].replace("rs_", "") for p in self.parked}
        for k, reg in self.side.items():
            it = self.it[k]; mine = self.objs[k]
            others = self.robot + [g for kk, gs in self.objs.items() if kk != k and any(kk in o for o in out) for g in gs]
            need = float(reg.get("clearance_m", 0.04))
            for _ in range(300):
                xy = self.r_xy + self.r_fwd * rng.uniform(*reg["fwd"]) + self.r_left * rng.uniform(*reg["left"])
                q = _qmul(_yaw_q(rng.uniform(0, 2 * np.pi)), it["q0"][3:7])
                data.qpos[it["adr"]:it["adr"] + 2] = xy; data.qpos[it["adr"] + 2] = it["q0"][2]
                data.qpos[it["adr"] + 3:it["adr"] + 7] = q / np.linalg.norm(q); data.qvel[it["dof"]:it["dof"] + 6] = 0
                mj.mj_forward(self.m, data)
                pts = np.vstack([self._geom_points(data, it["body"])])
                tf = np.array([self._to_table(p) for p in pts[:, :2]])
                if np.any(np.abs(tf) > self.half - 0.02):
                    continue                              # hangs off the table
                if min(mj.mj_geomDistance(self.m, data, u, o, 0.3, fr) for u in mine for o in others) >= need:
                    break
            else:
                data.qpos[it["adr"]:it["adr"] + 7] = it["q0"]; mj.mj_forward(self.m, data)
                print(f"[layout] no free spot for {it['name']} in its region; built spot")

    def apply(self, data, rng=None):
        rng = rng or self.rng
        self.park(data)
        for k in self.fixed:
            it = self.it[k]; data.qpos[it["adr"]:it["adr"] + 7] = it["q0"]; data.qvel[it["dof"]:it["dof"] + 6] = 0
        # one coin per reset: retrying a rejected jitter must not re-flip it,
        # or the tighter of the two arrangements ends up under-sampled
        swap = rng.uniform() < self.cfg["swap_prob"]
        for tries in range(300):
            s = self._sample(rng, swap)
            if s is None:
                continue
            swap = self._write(data, s, rng)
            self.mj.mj_forward(self.m, data)
            if self._robot_gap(data) >= self.cfg["robot_clearance_m"]:
                break
        else:
            print("[layout] no layout clear of the robot in 300 tries; using the built one")
            for it in self.items:
                data.qpos[it["adr"]:it["adr"] + 7] = it["q0"]
            self.mj.mj_forward(self.m, data)
            self._place_side(data, rng)
            return {}
        self._place_side(data, rng)
        print(f"[layout] {'SWAPPED ' if swap else ''}plate {np.round(s[1], 3)} "
              f"basket {np.round(s[2], 3)} (table frame), basket yaw "
              f"{np.degrees(s[3]):+.0f} deg", flush=True)
        # what was drawn, for the per-episode layout log (run_mujoco_wbc_sim.py)
        self.last = {"swapped": bool(swap),
                     "plate_table_xy": np.round(s[1], 4).tolist(),
                     "basket_table_xy": np.round(s[2], 4).tolist(),
                     "basket_yaw_deg": round(float(np.degrees(s[3])), 2),
                     "fruit_on_plate_xy": {self.it[k]["name"]: np.round(v, 4).tolist()
                                           for k, v in s[4].items()}}
        return {it["name"]: data.qpos[it["adr"]:it["adr"] + 2].copy() for it in self.items}

    def table_frame(self, xy):
        """World xy -> (along the table, across it) in m, the frame `last` uses."""
        return self._to_table(xy)

    def zero_velocities(self, model, data):
        for it in self.items:
            data.qvel[it["dof"]:it["dof"] + 6] = 0.0
