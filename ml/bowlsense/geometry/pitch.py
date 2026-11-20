"""Pitch calibration: 4 clicked crease corners -> homography between image and pitch metres.

Pitch frame (metres): x along the pitch from the bowler's popping crease (0) towards the
batter's popping crease (``crease_length_m``, 17.68 m = 20.12 m between stumps less 2 x 1.22 m);
y across the pitch, positive to the bowler's left, 0 on the stump line. The 4 points are the
ends of both popping creases inside the return creases, in this order (as seen by the
bowler): bowler-end left, bowler-end right, batter-end right, batter-end left.

The ground homography is only valid for points on the ground (feet, the ball at bounce). An
airborne ball is placed on the pitch's centre line instead (``along_pitch``): for a side-on
camera, the centre-line point with the same image column.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.errors import CalibrationError
from bowlsense.geometry.angles import FloatArray

POPPING_TO_STUMPS_M = 1.22
STUMPS_TO_STUMPS_M = 20.12
CREASE_LENGTH_M = STUMPS_TO_STUMPS_M - 2 * POPPING_TO_STUMPS_M  # 17.68 m
CREASE_WIDTH_M = 2.64  # between the return creases (popping crease half-width 1.32 m)

Point = Sequence[float]


class PitchCalibration(BaseModel):
    """A solved calibration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image_points: list[tuple[float, float]]
    image_to_pitch: list[list[float]]  # 3x3 homography
    pitch_to_image: list[list[float]]

    @property
    def h(self) -> FloatArray:
        """Image -> pitch homography as an array."""
        return np.asarray(self.image_to_pitch, dtype=np.float64)

    @property
    def h_inv(self) -> FloatArray:
        """Pitch -> image homography as an array."""
        return np.asarray(self.pitch_to_image, dtype=np.float64)


def pitch_corners() -> FloatArray:
    """The 4 reference corners in pitch metres, in the clicking order."""
    half = CREASE_WIDTH_M / 2.0
    return np.array(
        [[0.0, half], [0.0, -half], [CREASE_LENGTH_M, -half], [CREASE_LENGTH_M, half]],
        dtype=np.float64,
    )


def calibrate(points: Sequence[Point]) -> PitchCalibration:
    """Solve the homography from 4 image points (see module docstring for the order).

    Raises:
        CalibrationError: for the wrong number of points, repeated or collinear points, or
            a degenerate result.
    """
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise CalibrationError("Calibration needs exactly 4 points, each [x, y] in pixels.")
    img = np.asarray(points, dtype=np.float64)
    if not np.isfinite(img).all():
        raise CalibrationError("Calibration points must be finite numbers.")
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(img[i] - img[j]) < 2.0:
                raise CalibrationError("Two calibration points are (almost) the same.")
    for tri in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b, c = img[list(tri)]
        u, v = b - a, c - a
        if abs(float(u[0] * v[1] - u[1] * v[0])) < 1.0:
            raise CalibrationError("Three calibration points lie on one line; click the 4 corners.")
    h, _ = cv2.findHomography(img, pitch_corners())
    if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise CalibrationError("Could not solve the pitch calibration from these points.")
    return PitchCalibration(
        image_points=[(float(x), float(y)) for x, y in img],
        image_to_pitch=h.tolist(),
        pitch_to_image=np.linalg.inv(h).tolist(),
    )


def _apply(h: FloatArray, pts: FloatArray) -> FloatArray:
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    hom = np.hstack([p, np.ones((len(p), 1))]) @ h.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(hom[:, :2] / hom[:, 2:3], dtype=np.float64)


def image_to_pitch(cal: PitchCalibration, pts: FloatArray) -> FloatArray:
    """Image pixels -> pitch metres, for points ON THE GROUND."""
    return _apply(cal.h, pts)


def pitch_to_image(cal: PitchCalibration, pts: FloatArray) -> FloatArray:
    """Pitch metres -> image pixels."""
    return _apply(cal.h_inv, pts)


def along_pitch(cal: PitchCalibration, image_x: FloatArray, samples: int = 2001) -> FloatArray:
    """Distance along the pitch (m) of airborne points from their image column.

    The centre line (y = 0) is sampled from 3 m behind the bowler's crease to 3 m past the
    batter's, and each image column is matched to the centre-line point that projects to it
    (monotonic for a side-on camera). ``NaN`` outside that span or when the centre line isn't
    monotonic in the image (not a side-on view).
    """
    xs = np.linspace(-3.0, CREASE_LENGTH_M + 3.0, samples)
    cols = pitch_to_image(cal, np.stack([xs, np.zeros_like(xs)], axis=1))[:, 0]
    diffs = np.diff(cols)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        return np.full(np.shape(image_x), np.nan)
    order = np.argsort(cols)
    u = np.asarray(image_x, dtype=np.float64)
    out = np.interp(u, cols[order], xs[order], left=np.nan, right=np.nan)
    return np.asarray(out, dtype=np.float64)


def is_side_on(cal: PitchCalibration, min_span_px: float = 50.0) -> bool:
    """True when the pitch runs across the image (side-on camera), so ``along_pitch`` works."""
    ends = pitch_to_image(cal, np.array([[0.0, 0.0], [CREASE_LENGTH_M, 0.0]]))
    span = abs(float(ends[1, 0] - ends[0, 0]))
    return span >= min_span_px and bool(np.isfinite(along_pitch(cal, ends[:, 0])).all())
