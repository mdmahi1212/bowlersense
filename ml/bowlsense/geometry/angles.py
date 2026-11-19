"""Joint angles, segment angles, velocities and keypoint array helpers.

Conventions:
- Points are NumPy arrays of shape ``(..., 2)`` or ``(..., 3)`` in image pixels (x right, y down).
- Missing keypoints are ``NaN``; every function propagates ``NaN`` instead of raising.
- Angles are degrees. Segment angles are measured above the horizontal with *up* positive,
  even though image y points down.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bowlsense.errors import InvalidInputError
from bowlsense.schemas import KEYPOINT_NAMES, AngleSeries, PoseSequence

FloatArray = NDArray[np.float64]

NAME_INDEX: dict[str, int] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}


def _as_float(a: object) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


def joint_angle(a: object, b: object, c: object) -> FloatArray:
    """Interior angle ABC at vertex ``b`` in degrees (0-180; 180 = straight).

    Uses a clipped ``arccos`` so collinear points never produce ``NaN``. Returns ``NaN`` where
    any point is missing or a segment has zero length.
    """
    pa, pb, pc = _as_float(a), _as_float(b), _as_float(c)
    u, v = pa - pb, pc - pb
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = nu * nv
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / denom
    cos = np.where(denom > 1e-12, cos, np.nan)
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), dtype=np.float64)


def segment_angle(start: object, end: object) -> FloatArray:
    """Angle of the segment start->end above the image horizontal, in (-180, 180].

    0 = pointing right, 90 = pointing up the image, -90 = pointing down.
    """
    p0, p1 = _as_float(start), _as_float(end)
    dx = p1[..., 0] - p0[..., 0]
    dy_up = -(p1[..., 1] - p0[..., 1])
    out = np.degrees(np.arctan2(dy_up, dx))
    out = np.where(out <= -180.0, 180.0, out)  # -0.0 in y gives -180; keep (-180, 180]
    zero = (np.abs(dx) < 1e-12) & (np.abs(dy_up) < 1e-12)
    return np.asarray(np.where(zero, np.nan, out), dtype=np.float64)


def elevation_angle(start: object, end: object) -> FloatArray:
    """Angle of start->end above the horizontal regardless of left/right, in [-90, 90]."""
    p0, p1 = _as_float(start), _as_float(end)
    dx = np.abs(p1[..., 0] - p0[..., 0])
    dy_up = -(p1[..., 1] - p0[..., 1])
    out = np.degrees(np.arctan2(dy_up, dx))
    zero = (dx < 1e-12) & (np.abs(dy_up) < 1e-12)
    return np.asarray(np.where(zero, np.nan, out), dtype=np.float64)


def line_tilt(left: object, right: object) -> FloatArray:
    """Tilt of a body line (e.g. shoulders) from horizontal in (-90, 90], independent of
    which end is listed first: positive when the second point is higher."""
    ang = segment_angle(left, right)
    return np.asarray(((ang + 90.0) % 180.0) - 90.0, dtype=np.float64)


def _check_fps(fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise InvalidInputError(f"fps must be a positive number, got {fps}.")


def angular_velocity(angles_deg: object, fps: float) -> FloatArray:
    """Angular velocity in deg/s from an angle series, unwrapping jumps across +-180.

    ``NaN`` samples stay ``NaN``; unwrapping and differentiation run within each contiguous
    run of valid samples only.
    """
    _check_fps(fps)
    a = _as_float(angles_deg)
    out = np.full(a.shape, np.nan)
    for start, stop in valid_runs(np.isfinite(a)):
        seg = np.unwrap(np.radians(a[start:stop]))
        if seg.size >= 2:
            out[start:stop] = np.degrees(np.gradient(seg)) * fps
    return out


def linear_speed(points: object, fps: float) -> FloatArray:
    """Speed in pixels/s of a point track of shape ``(frames, 2|3)``; ``NaN`` where missing."""
    _check_fps(fps)
    p = _as_float(points)
    if p.ndim != 2:
        raise InvalidInputError(f"points must have shape (frames, dims), got {p.shape}.")
    valid = np.all(np.isfinite(p), axis=1)
    out = np.full(p.shape[0], np.nan)
    for start, stop in valid_runs(valid):
        if stop - start >= 2:
            vel = np.gradient(p[start:stop], axis=0) * fps
            out[start:stop] = np.linalg.norm(vel, axis=1)
    return out


def velocity(values: object, fps: float) -> FloatArray:
    """Time derivative (units/s) of a 1-D or ``(frames, dims)`` series, per valid run."""
    _check_fps(fps)
    v = _as_float(values)
    valid = np.isfinite(v) if v.ndim == 1 else np.all(np.isfinite(v), axis=1)
    out = np.full(v.shape, np.nan)
    for start, stop in valid_runs(valid):
        if stop - start >= 2:
            out[start:stop] = np.gradient(v[start:stop], axis=0) * fps
    return out


def valid_runs(mask: object) -> list[tuple[int, int]]:
    """``[start, stop)`` index ranges of consecutive ``True`` values."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise InvalidInputError("valid_runs expects a 1-D mask.")
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def keypoint_array(seq: PoseSequence) -> tuple[FloatArray, FloatArray]:
    """All keypoints as arrays: ``xy`` of shape ``(frames, 21, 2)`` and ``conf`` of shape
    ``(frames, 21)``. Missing keypoints are ``NaN`` in ``xy`` and ``0`` in ``conf``."""
    n = len(seq.frames)
    xy = np.full((n, len(KEYPOINT_NAMES), 2), np.nan)
    conf = np.zeros((n, len(KEYPOINT_NAMES)))
    for f, frame in enumerate(seq.frames):
        for name, kp in frame.keypoints.items():
            if kp is not None:
                i = NAME_INDEX[name]
                xy[f, i] = (kp.x, kp.y)
                conf[f, i] = kp.conf
    return xy, conf


def keypoint_track(seq: PoseSequence, name: str) -> FloatArray:
    """One keypoint over time, shape ``(frames, 2)``; ``NaN`` where missing."""
    if name not in NAME_INDEX:
        raise InvalidInputError(f"Unknown keypoint name '{name}'.")
    out = np.full((len(seq.frames), 2), np.nan)
    for f, frame in enumerate(seq.frames):
        kp = frame.get(name)
        if kp is not None:
            out[f] = (kp.x, kp.y)
    return out


def midpoint(a: object, b: object) -> FloatArray:
    """Midpoint of two point tracks (``NaN`` if either is missing)."""
    return np.asarray((_as_float(a) + _as_float(b)) / 2.0, dtype=np.float64)


def to_angle_series(name: str, values: object, frames: list[int] | None = None) -> AngleSeries:
    """Wrap an angle array as an ``AngleSeries`` (``NaN`` becomes ``None``)."""
    v = _as_float(values)
    idx = list(range(v.shape[0])) if frames is None else frames
    if len(idx) != v.shape[0]:
        raise InvalidInputError("frames and values must have the same length.")
    return AngleSeries(
        name=name,
        values=[None if not np.isfinite(x) else float(x) for x in v],
        frames=idx,
    )
