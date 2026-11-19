"""Keypoint cleaning: confidence gating, short-gap interpolation, Savitzky-Golay smoothing.

Per keypoint and coordinate:
1. Drop samples with confidence below ``pose.min_conf``.
2. Linearly interpolate interior gaps of at most ``pose.max_gap_frames`` frames. Longer gaps,
   and gaps at the start or end of the clip, stay missing.
3. Smooth each contiguous run of samples with a Savitzky-Golay filter (window about
   ``fps * pose.savgol_window_s`` frames, odd). Runs are smoothed separately, so the filter
   never reaches across a long gap.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from bowlsense.config import PoseSettings
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.angles import NAME_INDEX, valid_runs
from bowlsense.schemas import KEYPOINT_NAMES, FramePose, Keypoint, PoseSequence

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def savgol_window(fps: float, window_s: float, polyorder: int) -> int:
    """Odd Savitzky-Golay window length for a frame rate (at least ``polyorder + 2``)."""
    if fps <= 0 or window_s <= 0:
        raise InvalidInputError("fps and window_s must be positive.")
    w = round(fps * window_s)
    if w % 2 == 0:
        w += 1
    smallest = polyorder + 2 if polyorder % 2 == 1 else polyorder + 1
    return max(w, smallest)


def fill_short_gaps(values: FloatArray, valid: BoolArray, max_gap: int) -> BoolArray:
    """Linearly interpolate interior gaps of length ``<= max_gap`` in place.

    Returns the new validity mask. Edge gaps are never extrapolated.
    """
    filled = valid.copy()
    known = np.flatnonzero(valid)
    if known.size < 2 or max_gap <= 0:
        return filled
    for left, right in pairwise(known):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            span = np.arange(1, gap + 1) / (gap + 1)
            values[left + 1 : right] = values[left] + span * (values[right] - values[left])
            filled[left + 1 : right] = True
    return filled


def _gap_conf(conf: FloatArray, valid: BoolArray, filled: BoolArray) -> FloatArray:
    """Confidence for interpolated samples: the lower of the two neighbouring real samples."""
    out = np.where(valid, conf, 0.0)
    known = np.flatnonzero(valid)
    for left, right in pairwise(known):
        if right - left > 1 and filled[left + 1 : right].all():
            out[left + 1 : right] = min(conf[left], conf[right])
    return out


def smooth_runs(values: FloatArray, valid: BoolArray, window: int, polyorder: int) -> FloatArray:
    """Savitzky-Golay smoothing applied separately to each contiguous valid run.

    Runs shorter than the window use the largest odd window that fits; runs too short for
    the polynomial order are left as they are.
    """
    out = values.copy()
    for start, stop in valid_runs(valid):
        length = stop - start
        w = min(window, length if length % 2 == 1 else length - 1)
        if w <= polyorder:
            continue
        out[start:stop] = savgol_filter(values[start:stop], w, polyorder)
    return out


def clean_sequence(seq: PoseSequence, settings: PoseSettings) -> PoseSequence:
    """Return a cleaned copy of ``seq`` (gated, gap-filled and smoothed).

    Missing keypoints that couldn't be filled come back as ``None``. ``z`` is smoothed too
    when the input has it.
    """
    n = len(seq.frames)
    k = len(KEYPOINT_NAMES)
    xyz = np.full((n, k, 3), np.nan)
    conf = np.zeros((n, k))
    for f, frame in enumerate(seq.frames):
        for name, kp in frame.keypoints.items():
            if kp is None:
                continue
            j = NAME_INDEX[name]
            xyz[f, j] = (kp.x, kp.y, np.nan if kp.z is None else kp.z)
            conf[f, j] = kp.conf

    window = savgol_window(seq.fps, settings.savgol_window_s, settings.savgol_polyorder)
    out_xyz = np.full_like(xyz, np.nan)
    out_valid = np.zeros((n, k), dtype=bool)
    out_conf = np.zeros((n, k))
    for j in range(k):
        valid = np.isfinite(xyz[:, j, 0]) & np.isfinite(xyz[:, j, 1])
        valid &= conf[:, j] >= settings.min_conf
        filled = valid
        for d in range(3):
            col = np.where(valid, xyz[:, j, d], np.nan)
            if d == 2 and not np.isfinite(col[valid]).all():
                continue  # no (or partial) depth: leave z missing
            filled = fill_short_gaps(col, valid, settings.max_gap_frames)
            out_xyz[:, j, d] = smooth_runs(col, filled, window, settings.savgol_polyorder)
        if not valid.any():
            continue
        filled = fill_short_gaps(np.zeros(n), valid, settings.max_gap_frames)
        out_valid[:, j] = filled
        out_conf[:, j] = _gap_conf(conf[:, j], valid, filled)

    frames: list[FramePose] = []
    for f, frame in enumerate(seq.frames):
        kps: dict[str, Keypoint | None] = {}
        for name in KEYPOINT_NAMES:
            j = NAME_INDEX[name]
            if not out_valid[f, j]:
                kps[name] = None
                continue
            z = out_xyz[f, j, 2]
            kps[name] = Keypoint(
                x=float(out_xyz[f, j, 0]),
                y=float(out_xyz[f, j, 1]),
                z=float(z) if np.isfinite(z) else None,
                conf=float(np.clip(out_conf[f, j], 0.0, 1.0)),
            )
        frames.append(FramePose(frame_idx=frame.frame_idx, t=frame.t, keypoints=kps))
    return seq.model_copy(update={"frames": frames})
