"""Randomize the table-top object layout on every sim reset.

Data collection wants the apple and the basket somewhere different each
episode, but not somewhere the teleoperator cannot comfortably reach.  So the
sample region is the intersection of three constraints, all derived from the
model at runtime (no scene-frame constants - the WBC scene and the render
scene use different world frames):

  * the table top disc, inset by each object's own footprint radius
  * the half of that disc facing the robot ("divide the table in half")
  * an annulus of reachable radii around the robot's base

Objects are also kept apart so the apple never spawns inside the basket.
"""

from __future__ import annotations

import numpy as np

TOP_EPS = 0.002          # lift above the table top so nothing starts interpenetrating
DEFAULT_REACH = (0.20, 0.42)
# Signed angle window around the robot's own forward axis, degrees,
# positive = the robot's left. Measured from the 145 recorded demos:
# grasps landed at -65..+11 deg, releases at -50..+14. This is that
# envelope with margin, which keeps every sample inside the right
# arm's worked range while spanning far more of the table than the
# old near-half rule did.
DEFAULT_SECTOR = (-80.0, 45.0)


def _body_aabb(model, data, body):
    import mujoco
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body:
            continue
        pos, mat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            v0, n0 = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            V = model.mesh_vert[v0:v0 + n0].reshape(-1, 3) @ mat.T + pos
            lo, hi = np.minimum(lo, V.min(0)), np.maximum(hi, V.max(0))
        else:
            r = float(model.geom_size[g].max())
            lo, hi = np.minimum(lo, pos - r), np.maximum(hi, pos + r)
    return lo, hi


def _find_body(model, needle):
    import mujoco
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if needle in nm:
            return b, nm
    return -1, ""


class TableRandomizer:
    """Resample the free-joint poses of table-top objects.

    Built once against a forwarded MjData in the *nominal* layout - that pass
    is what measures the table and each object's footprint.
    """

    def __init__(self, model, data, objects=("apple", "basket"),
                 table="table", base="base", reach=DEFAULT_REACH,
                 margin=0.02, seed=None, reach_overrides=None,
                 sector=DEFAULT_SECTOR,
                 avoid=(), avoid_clearance=0.01):
        import mujoco
        self.rng = np.random.default_rng(seed)
        self.reach = tuple(reach)
        self.sector = None if sector is None else tuple(float(v) for v in sector)
        # {needle: (lo, hi)} -- lets one object sit nearer the robot than the
        # rest. Without it every object shares one annulus, so pulling the apple
        # in would drag the basket in with it.
        self.reach_overrides = {k: tuple(v)
                                for k, v in (reach_overrides or {}).items()}
        self.margin = float(margin)

        tb, tname = _find_body(model, table)
        if tb < 0:
            raise ValueError(f"no body matching {table!r}")
        lo, hi = _body_aabb(model, data, tb)
        self.top_z = float(hi[2])
        self.center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
        self.radius = float(min(hi[0] - lo[0], hi[1] - lo[1]) / 2)

        # The robot's base free joint fixes which half of the table is "near".
        self.base_xy = None
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and base in nm:
                self.base_xy = data.qpos[model.jnt_qposadr[j]:
                                         model.jnt_qposadr[j] + 2].copy()
                break
        if self.base_xy is None:
            raise ValueError(f"no free joint matching {base!r}")
        u = self.center - self.base_xy
        n = float(np.linalg.norm(u))
        # +u points from the robot toward the table centre, so the near half is
        # where the offset from centre projects negative onto u.
        self.toward = u / n if n > 1e-9 else np.array([1.0, 0.0])

        # The sector is measured off the robot's OWN heading, not off the
        # direction to the table centre - that is the frame the demos' grasp
        # azimuths live in, so the numbers transfer directly.
        self.fwd, self.left = self.toward, np.array([-self.toward[1],
                                                     self.toward[0]])
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and base in nm:
                adr = model.jnt_qposadr[j]
                w, x, y, z = data.qpos[adr + 3:adr + 7]
                # body x-axis in world, dropped to the plane
                fx, fy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
                nf = float(np.hypot(fx, fy))
                if nf > 1e-9:
                    self.fwd = np.array([fx, fy]) / nf
                    self.left = np.array([-self.fwd[1], self.fwd[0]])
                break

        self.items = []
        for needle in objects:
            b, name = _find_body(model, needle)
            if b < 0:
                raise ValueError(f"no body matching {needle!r}")
            j = int(model.body_jntadr[b])
            if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"body {name!r} has no free joint")
            olo, ohi = _body_aabb(model, data, b)
            adr = int(model.jnt_qposadr[j])
            nominal = data.qpos[adr:adr + 7].copy()
            foot = float(max(ohi[0] - olo[0], ohi[1] - olo[1]) / 2)
            self.items.append({
                "name": name,
                "reach": self.reach_overrides.get(needle, self.reach),
                "near_only": True,
                # Keep-out uses the footprint's short half-axis, not `foot`.
                # `foot` is the circumscribed radius - demanding that much
                # around every hand leaves the basket nowhere to go but the far
                # side of the table, out of range for dropping the apple in.
                # What actually matters is that no hand sits inside the object.
                "gap": float(min(ohi[0] - olo[0], ohi[1] - olo[1]) / 2)
                       + float(avoid_clearance),
                "adr": adr,
                "quat": nominal[3:7].copy(),
                "foot": foot,
                # keep the object's own resting height above the table top
                "lift": float(nominal[2] - olo[2]),
            })

        # At the reset stance the palms hover over the very annulus the sampler
        # draws from, so whatever lands there gets shoved by the first arm
        # motion.  Exclude a disc around every hand and wrist link, measured
        # from this same forwarded data - still no scene-frame constants.
        self.avoid_clearance = float(avoid_clearance)
        keepout = []
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not any(k in nm for k in avoid):
                continue
            klo, khi = _body_aabb(model, data, b)
            if np.isfinite(klo).all():
                keepout.append((klo[:2] + khi[:2]) / 2)
        self.keepout = np.array(keepout) if keepout else None

        # Anything else resting on the table with a free joint - the fruits, in
        # scenes that have them - is not ours to move, but it is very much in
        # the way. Without this the apple happily respawns inside a watermelon.
        self.obstacles = []
        own = {it["adr"] for it in self.items}
        for b in range(model.nbody):
            j = int(model.body_jntadr[b])
            if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            if int(model.jnt_qposadr[j]) in own:
                continue
            olo, ohi = _body_aabb(model, data, b)
            if not np.isfinite(olo).all():
                continue
            # On the table, not the floor and not part of the robot.
            if not (self.top_z - 0.02 <= olo[2] <= self.top_z + 0.15):
                continue
            if np.linalg.norm((olo[:2] + ohi[:2]) / 2 - self.center) > self.radius:
                continue
            self.obstacles.append(((olo[:2] + ohi[:2]) / 2,
                                   float(max(ohi[0] - olo[0], ohi[1] - olo[1]) / 2)))

        # A wide object can have nowhere left inside its nominal band once the
        # hands are excluded - the basket is exactly that case, its footprint is
        # 0.27 m across and the hands straddle the near half.  Rather than give
        # up and hand back no layout at all, push its outer radius out, and only
        # if that still fails let it use the far half of the table.
        for it in self.items:
            self._fit_reach(it)

    # ------------------------------------------------------------------
    def _clear_of_hands(self, xy, gap):
        if self.keepout is None or gap is None:
            return True
        return bool(np.all(np.linalg.norm(self.keepout - xy, axis=1) >= gap))

    def _fit_reach(self, it, tries=2000, want=20):
        """Widen one item's sample region until the hands stop blocking it.

        Near half first, all the way out, before ever considering the far one -
        an object the robot cannot comfortably reach is worse than one that
        occasionally starts near a hand.
        """
        lo_r, hi_r = it["reach"]
        limit = float(np.linalg.norm(self.center - self.base_xy)) + self.radius
        rng = np.random.default_rng(0)
        for near_only in (True, False):
            hi = hi_r
            while hi <= limit + 1e-9:
                hits = sum(self._sample_xy(it["foot"], rng, (lo_r, hi),
                                           near_only, tries=1,
                                           gap=it["gap"]) is not None
                           for _ in range(tries))
                if hits >= want:
                    it["reach"] = (lo_r, hi)
                    it["near_only"] = near_only
                    return
                hi += 0.03
        it["reach"] = (lo_r, limit)            # nothing fits - fall back to the
        it["near_only"] = False                # old unconstrained behaviour

    # ------------------------------------------------------------------
    def _sample_xy(self, foot, rng, reach=None, near_only=True, tries=400,
                   gap=None):
        """One point on the table the robot can reach, clear of its hands."""
        r_max = self.radius - foot - self.margin
        lo_r, hi_r = reach if reach is not None else self.reach
        for _ in range(tries):
            a = rng.uniform(0, 2 * np.pi)
            r = r_max * np.sqrt(rng.uniform())
            off = np.array([np.cos(a), np.sin(a)]) * r
            if self.sector is None and near_only and off @ self.toward > 0:
                off -= 2 * (off @ self.toward) * self.toward   # far half, flipped back
            xy = self.center + off
            v = xy - self.base_xy
            d = float(np.linalg.norm(v))
            if not (lo_r <= d <= hi_r):
                continue
            if self.sector is not None:
                ang = np.degrees(np.arctan2(v @ self.left, v @ self.fwd))
                if not (self.sector[0] <= ang <= self.sector[1]):
                    continue
            if self._clear_of_hands(xy, gap):
                return xy
        return None

    def apply(self, data, rng=None):
        """Write fresh poses into data.qpos. Returns a {name: xy} dict."""
        rng = rng or self.rng
        # Largest footprint first: it has the least room, so letting it choose
        # before the small objects crowd the disc keeps the retry count down.
        order = sorted(range(len(self.items)),
                       key=lambda i: -self.items[i]["foot"])
        # Two passes: honour the fruits if the table has room for them, fall
        # back to ignoring them if it does not. Without the fallback a crowded
        # table makes the search unsolvable, and apply() then grinds through
        # every retry before returning nothing - which reads to the user as
        # "randomization silently switched itself off".
        xys = None
        # 4 rounds, not 25, on the obstacle pass: when the table has room it
        # succeeds on the first, and when it has not, 25 doomed rounds cost a
        # couple of seconds on every single reset.
        for obstacles, rounds in ((self.obstacles, 4), ([], 25)):
          for _ in range(rounds):              # resample the layout, not one object
            placed, trial = list(obstacles), {}
            for i in order:
                it = self.items[i]
                for _ in range(200):
                    cand = self._sample_xy(it["foot"], rng, it["reach"],
                                           it["near_only"], gap=it["gap"])
                    if cand is None:
                        break
                    if all(np.linalg.norm(cand - p) >= it["foot"] + f + self.margin
                           for p, f in placed):
                        placed.append((cand, it["foot"]))
                        trial[i] = cand
                        break
                else:
                    break
                if i not in trial:
                    break
            if len(trial) == len(self.items):
                xys = trial
                break
          if xys is not None:
            if not obstacles and self.obstacles:
                print("[randomize] table too crowded to avoid the static "
                      "objects; placing without them")
            break
        if xys is None:                        # unreachable geometry - leave as is
            return {}

        out = {}
        for i, it in enumerate(self.items):
            xy = xys[i]
            out[it["name"]] = xy
            yaw = rng.uniform(-np.pi, np.pi)
            qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            q = it["quat"]
            spun = np.array([
                qz[0] * q[0] - qz[3] * q[3],
                qz[0] * q[1] - qz[3] * q[2],
                qz[0] * q[2] + qz[3] * q[1],
                qz[0] * q[3] + qz[3] * q[0],
            ])
            data.qpos[it["adr"]:it["adr"] + 2] = xy
            data.qpos[it["adr"] + 2] = self.top_z + it["lift"] + TOP_EPS
            data.qpos[it["adr"] + 3:it["adr"] + 7] = spun / np.linalg.norm(spun)
        return out

    def zero_velocities(self, model, data):
        import mujoco
        for it in self.items:
            b, _ = _find_body(model, it["name"])
            j = int(model.body_jntadr[b])
            v = int(model.jnt_dofadr[j])
            data.qvel[v:v + 6] = 0.0
