#!/usr/bin/env python3
"""Render a wrist camera through an equidistant fisheye instead of a pinhole.

The real wrist cameras stream RAW fisheye: the robot's camera-server config
gives `hand_left` / `hand_right` as 1280x720, crop 1.0, and no `undistort` key,
so nothing rectifies them before they reach the VLA. A MuJoCo camera is always a
rectilinear pinhole, and a pinhole cannot imitate a fisheye by widening its fovy
- the two projections disagree more the further from centre you look. On these
lenses (~121 deg horizontal) a pinhole would have to spend 1077 px to reach the
angle the real lens puts at 320 px, so the forearm, which lives at the edge of
the frame, comes out ~3x too big in sim. That is the gap this closes.

Same construction as scripts/fisheye_render.py, which this imports rather than
duplicating: render the scene as cube faces from the wrist camera's own pose and
remap them into r = f*theta, the standard equidistant model for this class of
lens. f comes from the horizontal FOV, which is the number lens makers quote.

Two differences from the ELP eyes, both because 121 deg is much narrower than
170 deg:
  * only the faces that actually claim pixels are rendered (3 of 5 at 121 deg -
    front, left, right - because the fisheye never reaches 45 deg vertically),
  * the per-face remap tables are built once, at construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fisheye_render import FACES, fisheye_rays  # noqa: E402


class WristFisheye:
    """Remap tables for one fisheye lens, shared by every camera using it.

    The tables depend only on (width, height, fov_h, face_size), so both wrists
    share one instance; only the render pose differs.
    """

    def __init__(self, width: int, height: int, fov_h_deg: float,
                 face_size: int = 0):
        dirs, valid, f_px, vert = fisheye_rays(width, height, fov_h_deg)
        self.width, self.height = width, height
        self.fov_h, self.fov_v = float(fov_h_deg), float(vert)
        self.f_px = float(f_px)

        # A 90 deg face is f_face = S/2 px/rad at its centre, so S = 2*f keeps
        # the fisheye's own centre sampling; anything less resamples upwards and
        # softens the image. Rounded up to a multiple of 32 for the GL buffer.
        if face_size <= 0:
            face_size = int(np.ceil(2 * f_px / 32.0)) * 32
        self.face_size = int(face_size)
        S = self.face_size
        f_face = S / 2.0

        claimed = np.zeros((height, width), bool)
        self.faces = []
        for name, M in FACES.items():
            df = dirs @ M                                # eye frame -> face frame
            fwd = -df[:, :, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                xs, ys = df[:, :, 0] / fwd, df[:, :, 1] / fwd
            take = (valid & ~claimed & (fwd > 1e-6)
                    & (np.abs(xs) <= 1) & (np.abs(ys) <= 1))
            if not take.any():
                continue                                 # 121 deg never sees up/down
            claimed |= take                              # first face wins a seam
            mx = np.nan_to_num(f_face * xs + S / 2).astype(np.float32)
            my = np.nan_to_num(S / 2 - f_face * ys).astype(np.float32)
            # Remapping only the rows this face touches is most of the warp cost
            # at 1280x720; the side faces are thin vertical wedges.
            rows = np.flatnonzero(take.any(1))
            cols = np.flatnonzero(take.any(0))
            sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            self.faces.append((name, M, take[sl], mx[sl], my[sl], sl))
        self.coverage = float(claimed.mean())
        self.inside_unfilled = float((valid & ~claimed).mean())

    def describe(self) -> str:
        return (f"equidistant r=f*theta, f={self.f_px:.1f} px/rad, "
                f"{self.fov_h:.2f} deg horizontal -> {self.fov_v:.1f} deg "
                f"vertical over {self.height} px, {self.width}x{self.height}, "
                f"{len(self.faces)} cube faces of {self.face_size}px "
                f"({', '.join(f[0] for f in self.faces)}), "
                f"{100 * self.coverage:.1f}% of the frame covered")

    def render(self, sensor, model, data, cam_id: int,
               probe_name: str, probe_id: int, cv2) -> np.ndarray:
        """Cube-render from the camera at `cam_id`, warped to fisheye.

        `sensor` must be built at face_size x face_size. The probe camera has to
        live on the worldbody: its pos/quat are world-frame there, so they can be
        pointed anywhere, which a body-mounted camera's cannot.
        """
        import mujoco

        eye_pos = data.cam_xpos[cam_id].copy()
        eye_R = data.cam_xmat[cam_id].reshape(3, 3).copy()
        S = self.face_size
        f_face = S / 2.0
        img = np.zeros((self.height, self.width, 3), np.uint8)

        for _name, M, take, mx, my, sl in self.faces:
            model.cam_pos[probe_id] = eye_pos
            R = eye_R @ M
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, R.reshape(9))
            model.cam_quat[probe_id] = q
            model.cam_fovy[probe_id] = 90.0
            # Only the camera frames need recomputing, not the whole dynamics -
            # mj_forward here would cost more than the warp.
            mujoco.mj_camlight(model, data)
            face = sensor.render(
                model, data, probe_name,
                camera_params={"position": data.cam_xpos[probe_id].copy(),
                               "rotation": data.cam_xmat[probe_id].reshape(3, 3).copy(),
                               "fx": f_face, "fy": f_face,
                               "width": S, "height": S})
            warped = cv2.remap(face, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            sub = img[sl]
            sub[take] = warped[take]
        return img
