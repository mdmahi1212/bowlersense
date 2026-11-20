"""Synthetic ball flight seen by a pinhole camera, for testing calibration and ball speed.

World frame (metres): X along the pitch from the bowler's popping crease, Y across (+ = the
bowler's left), Z up. Image: u right, v down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bowlsense.geometry.pitch import CREASE_LENGTH_M, CREASE_WIDTH_M

FloatArray = NDArray[np.float64]
G = 9.81
BALL_RADIUS_M = 0.036


@dataclass(frozen=True)
class Camera:
    """A pinhole camera looking from ``eye`` at ``target``."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_px: float = 1400.0
    width: int = 1280
    height: int = 720

    def project(self, pts: FloatArray) -> FloatArray:
        """World points (N, 3) -> image pixels (N, 2)."""
        eye, tgt = np.asarray(self.eye), np.asarray(self.target)
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        rel = np.atleast_2d(pts) - eye
        xc, yc, zc = rel @ right, rel @ down, rel @ fwd
        return np.stack(
            [self.width / 2 + self.focal_px * xc / zc, self.height / 2 + self.focal_px * yc / zc],
            axis=1,
        )

    def crease_points(self) -> FloatArray:
        """The 4 calibration corners (bowler-end left, right, batter-end right, left)."""
        half = CREASE_WIDTH_M / 2.0
        corners = [[0.0, half, 0.0], [0.0, -half, 0.0], [CREASE_LENGTH_M, -half, 0.0],
                   [CREASE_LENGTH_M, half, 0.0]]  # fmt: skip
        return self.project(np.array(corners))


SIDE_ON = Camera(eye=(CREASE_LENGTH_M / 2, -32.0, 1.6), target=(CREASE_LENGTH_M / 2, 0.0, 1.0))
ANGLED = Camera(eye=(4.0, -30.0, 2.0), target=(10.0, 0.0, 1.0))  # ~11 deg off side-on
BEHIND = Camera(eye=(-18.0, 0.0, 2.2), target=(10.0, 0.0, 0.5))


def flight(
    speed_kmh: float,
    fps: float,
    duration_s: float = 0.6,
    release: tuple[float, float, float] = (-0.3, 0.25, 2.2),
    down_deg: float = 7.0,
    lateral_deg: float = -0.6,
    restitution: float = 0.45,
) -> tuple[FloatArray, int]:
    """Ball centre (frames, 3) from release, and the index of the bounce frame."""
    v = speed_kmh / 3.6
    d, a = np.radians(down_deg), np.radians(lateral_deg)
    vel = np.array([v * np.cos(d) * np.cos(a), v * np.cos(d) * np.sin(a), -v * np.sin(d)])
    pos = np.array(release, dtype=np.float64)
    dt = 1.0 / fps
    out, bounce = [], -1
    for i in range(int(duration_s * fps)):
        out.append(pos.copy())
        pos = pos + vel * dt + np.array([0.0, 0.0, -0.5 * G * dt * dt])
        vel = vel + np.array([0.0, 0.0, -G * dt])
        if pos[2] < BALL_RADIUS_M and vel[2] < 0:
            pos[2] = 2 * BALL_RADIUS_M - pos[2]
            vel = np.array([vel[0] * 0.9, vel[1] * 0.9, -vel[2] * restitution])
            if bounce < 0:
                bounce = i + 1
    return np.array(out), bounce
