#!/usr/bin/env python3
"""Letterbox the MuJoCo interactive viewer down to a camera's PUBLISHED frame.

The problem this solves
-----------------------
Press ] in the interactive viewer to select `head_pinhole` and the picture does
not match the head_pinhole window that `vla_camera_server.py` publishes - the
viewer shows a much wider slice of the room. Both are the same camera, at the
same pose, with the same `fovy`, so this looks like a bug in one of them.

It is neither. MuJoCo's `fovy` is the VERTICAL field of view, and that is the
only angle a camera's frustum actually pins down. The horizontal half-angle is
derived at render time from the ASPECT RATIO OF THE VIEWPORT:

    mjv_updateCamera() fills mjvGLCamera.frustum_top/bottom from fovy and
    leaves frustum_width at 0; mjr_render() then takes the horizontal
    half-extent to be (top-bottom)/2 * viewport_width/viewport_height.

A camera's `resolution="768 432"` attribute does NOT enter that computation.
It only sizes an offscreen buffer. So an offscreen renderer at 768x432 and an
interactive window at, say, 1536x432 give the same vertical extent and the same
pixel scale, but the window shows twice the horizontal extent:

    head_pinhole, fovy = 51.084 deg
      published  768x432  (aspect 1.778)  ->  80.70 deg horizontal
      window    1536x432  (aspect 3.556)  -> 119.04 deg horizontal

The fix chosen here
-------------------
Draw four black quads a few centimetres in front of the selected camera, sized
so the hole between them subtends exactly the published frame's angles. The
scene outside the published frame is then hidden, and what is left in the
window is the published frame, at the correct aspect, undistorted.

Why not the alternatives:

  * Resize the window to the camera's aspect. The passive viewer owns its GLFW
    window inside C++; `Simulate.viewport` is read-only and there is no handle
    to pass to glfw, so Python cannot resize it. Doing it from outside needs
    xdotool/wmctrl (neither is installed here) and would still be defeated by
    the left/right UI panels, which change the 3D rect without changing the
    window.

  * Pin the horizontal angle by giving the camera a `sensorsize`+`focal`
    matching its declared resolution. This works - MuJoCo then sets
    frustum_width and ignores the viewport aspect - but it does NOT letterbox:
    the published frame is stretched anamorphically to fill the window
    (measured: a 1536x432 window shows the 768x432 frame squashed 2:1). A
    distorted picture is a bad tool for judging sim-vs-real.

  * Just draw an outline of the published frame. Least invasive, but it leaves
    the two pictures disagreeing and asks the eye to do the cropping.

Nothing here touches `fovy`, `resolution`, `pos` or `quat` of any camera, in
the MJCF or in memory: the mask is decor geometry, the camera intrinsics that
are calibrated against the real hardware are left exactly as the scene declares
them.

Caveat, stated plainly: the mask is geometry, so anything nearer to the lens
than MASK_NEAR_SCALE * the near clipping plane (~3 cm here) is hidden by it.
That is a centimetre beyond where MuJoCo already clips, so in practice only the
robot's own fingers at full close can reach it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# How far in front of the lens the mask sits, as a multiple of the near
# clipping plane. Must be > 1 or the mask is clipped away and invisible; the
# smaller it is, the less real geometry the mask can swallow.
MASK_NEAR_SCALE = 1.5

# How far past the published frame the quads extend, as a multiple of the
# window's own half-extent. 3 is generous enough that a very wide window, or a
# window the user reshapes while running, stays covered.
MASK_OVERSHOOT = 3.0

# Half-thickness of the mask slabs, metres. Thin so the front face is
# effectively at MASK_NEAR_SCALE * near.
MASK_THICKNESS = 1e-4

MASK_RGBA = (0.0, 0.0, 0.0, 1.0)


class PublishedFrameMask:
    """Mask the viewer down to the published frame of the selected camera.

    Usage, once per viewer.sync():

        mask.update(data, viewer.user_scn, viewer.cam, window_aspect)
    """

    def __init__(self, model, enabled: bool = True, overrides: dict | None = None):
        self.model = model
        self.enabled = enabled

        # near-plane distance in metres, the same number mjv_updateCamera uses
        self.near = float(model.vis.map.znear) * float(model.stat.extent)
        self.distance = MASK_NEAR_SCALE * self.near

        # camera id -> (published_w, published_h, tan(hfov/2), tan(vfov/2))
        self.frames: dict[int, tuple[int, int, float, float]] = {}

        for cid in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid) or ""
            w, h = (int(x) for x in model.cam_resolution[cid])

            if overrides and name in overrides:
                w, h = overrides[name]

            # resolution defaults to 1x1 when the MJCF does not declare one:
            # that camera has no published frame to agree with.
            if w <= 1 or h <= 1:
                continue

            tan_v = math.tan(math.radians(float(model.cam_fovy[cid])) / 2.0)
            self.frames[cid] = (w, h, tan_v * w / h, tan_v)

        self._last_cid = -2
        self._warned_narrow: set[int] = set()

    # ------------------------------------------------------------------

    def describe(self) -> str:
        """One line per camera the mask knows, with both fields of view."""

        rows = []

        for cid, (w, h, tan_h, tan_v) in sorted(self.frames.items()):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            rows.append(
                f"  {name:<22s} {w}x{h}  "
                f"{2 * math.degrees(math.atan(tan_h)):6.2f} deg H x "
                f"{2 * math.degrees(math.atan(tan_v)):6.2f} deg V"
            )

        if not rows:
            return "[frame] no camera in this scene declares a resolution"

        return ("[frame] the viewer is letterboxed to each camera's published "
                "frame:\n" + "\n".join(rows))

    # ------------------------------------------------------------------

    def update(self, data, user_scn, cam, window_aspect: float | None = None):
        """Rebuild the mask for whichever camera the viewer is showing."""

        if user_scn is None:
            return

        cid = (int(cam.fixedcamid)
               if int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_FIXED)
               else -1)

        if not self.enabled or cid not in self.frames:
            user_scn.ngeom = 0
            if cid != self._last_cid:
                self._last_cid = cid
                if self.enabled and cid >= 0:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
                    print(f"[frame] {name}: no resolution declared, so there is "
                          f"no published frame to match - showing the whole "
                          f"window", flush=True)
            return

        w, h, tan_h, tan_v = self.frames[cid]

        d = self.distance - MASK_THICKNESS
        hx, hy = d * tan_h, d * tan_v

        # The window's own frustum at the same depth: fovy fixes the vertical
        # half-extent, the viewport aspect sets the horizontal one.
        a_w = float(window_aspect) if window_aspect else (w / h)
        wx, wy = hy * a_w, hy

        if cid != self._last_cid:
            self._last_cid = cid
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            print(
                f"[frame] {name}: published {w}x{h} = "
                f"{2 * math.degrees(math.atan(tan_h)):.2f} deg H x "
                f"{2 * math.degrees(math.atan(tan_v)):.2f} deg V; the window "
                f"(aspect {a_w:.3f}) would show "
                f"{2 * math.degrees(math.atan(a_w * tan_v)):.2f} deg H - "
                f"masked to the published frame",
                flush=True,
            )

        if wx < hx * 0.999 and cid not in self._warned_narrow:
            self._warned_narrow.add(cid)
            print(f"[frame] the viewer window is narrower than this camera's "
                  f"{w}x{h} frame ({a_w:.3f} < {w / h:.3f}): the window itself "
                  f"crops the sides. Widen it to see the whole published frame.",
                  flush=True)

        user_scn.ngeom = self.paint(data, user_scn, cid, a_w)

    # ------------------------------------------------------------------

    def paint(self, data, scene, cid: int, window_aspect: float,
              start: int = 0) -> int:
        """Write the four mask quads into `scene` at `start`; return how many.

        Split out of update() so an offscreen mujoco.Renderer can be masked the
        same way the interactive viewer is - which is how this gets tested
        without opening a window.
        """

        w, h, tan_h, tan_v = self.frames[cid]

        d = self.distance

        # The quads have thickness, and it is their FRONT face that does the
        # occluding, so the hole has to be sized at d - t rather than d. Sizing
        # it at the slab's centre over-covers by t/d of the half-width, which
        # measured as a ~1 px dark border eating into the published frame.
        t = MASK_THICKNESS
        hx, hy = (d - t) * tan_h, (d - t) * tan_v
        wx, wy = hy * float(window_aspect), hy

        # Outer edge of the quads. Generous on purpose: the window can be
        # reshaped between syncs and an uncovered strip is worse than a wasted
        # one.
        X = max(wx, hx) * MASK_OVERSHOOT
        Y = max(wy, hy) * MASK_OVERSHOOT

        pos = np.asarray(data.cam_xpos[cid], dtype=float)
        mat = np.asarray(data.cam_xmat[cid], dtype=float).reshape(3, 3)

        # MuJoCo camera frame: +x right, +y up, -z is the viewing direction.
        def quad(cx, cy, sx, sy):
            return (
                pos + mat @ np.array([cx, cy, -d]),
                np.array([max(sx, 1e-6), max(sy, 1e-6), t]),
            )

        bars = [
            quad(-(hx + X) / 2.0, 0.0, (X - hx) / 2.0, Y),          # left
            quad(+(hx + X) / 2.0, 0.0, (X - hx) / 2.0, Y),          # right
            quad(0.0, +(hy + Y) / 2.0, hx, (Y - hy) / 2.0),         # top
            quad(0.0, -(hy + Y) / 2.0, hx, (Y - hy) / 2.0),         # bottom
        ]

        flat = mat.reshape(9)
        n = 0

        for p, size in bars:
            if start + n >= scene.maxgeom:
                break

            g = scene.geoms[start + n]

            mujoco.mjv_initGeom(
                g,
                int(mujoco.mjtGeom.mjGEOM_BOX),
                size.astype(np.float64),
                p.astype(np.float64),
                flat,
                np.asarray(MASK_RGBA, dtype=np.float32),
            )

            # Decor keeps it out of the shadow pass: a black slab 3 cm from the
            # lens would otherwise throw a shadow across the whole room.
            g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            g.emission = 1.0
            g.specular = 0.0
            g.shininess = 0.0
            g.reflectance = 0.0
            g.segid = -1

            n += 1

        return n


def parse_overrides(spec: str) -> dict:
    """'head_pinhole=768x432,elp_left=640x480' -> {'head_pinhole': (768, 432)...}"""

    out = {}

    for item in str(spec or "").replace(" ", "").split(","):
        if not item:
            continue

        name, _, wh = item.partition("=")
        w, _, h = wh.partition("x")

        if not (name and w.isdigit() and h.isdigit()):
            raise ValueError(f"bad camera frame {item!r}, want NAME=WxH")

        out[name] = (int(w), int(h))

    return out
