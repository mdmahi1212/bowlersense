"""Per-frame pose features for the learned models (phase/release model, delivery type).

Keypoints are made position- and scale-free: each joint relative to the hip midpoint, divided
by the bowler's torso length (median over the window), so camera distance and framing don't
matter. Bowling-side and front-side joints are reordered by role (bowling shoulder first,
etc.), so one model serves right- and left-arm bowlers. Missing joints become 0 with a
visibility channel saying so.
"""

from __future__ import annotations

import numpy as np

from bowlsense.geometry.angles import FloatArray, keypoint_array, midpoint
from bowlsense.geometry.body import side_of
from bowlsense.schemas import KEYPOINT_NAMES, PoseSequence

FEATURE_VERSION = 1
_PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe", "eye", "ear")
_CENTRE = ("nose",)


def role_order(arm: str) -> list[str]:
    """Keypoint names ordered by role: centre, then bowling side, then front side."""
    bowl = side_of(arm, "bowling")  # type: ignore[arg-type]
    front = side_of(arm, "front")  # type: ignore[arg-type]
    return [*_CENTRE, *(f"{bowl}_{p}" for p in _PARTS), *(f"{front}_{p}" for p in _PARTS)]


N_JOINTS = len(role_order("right"))
N_FEATURES = N_JOINTS * 5  # x, y, vx, vy, visible


def torso_length(xy: FloatArray, names: list[str]) -> float:
    """Median shoulder-midpoint to hip-midpoint distance (pixels) over the window."""
    idx = {n: i for i, n in enumerate(names)}
    sh = midpoint(xy[:, idx["l_shoulder"]], xy[:, idx["r_shoulder"]])
    hp = midpoint(xy[:, idx["l_hip"]], xy[:, idx["r_hip"]])
    d = np.linalg.norm(sh - hp, axis=1)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float("nan")


def frame_features(seq: PoseSequence) -> FloatArray:
    """``(frames, N_FEATURES)`` float32-ready array (see module docstring)."""
    xy, _ = keypoint_array(seq)
    names = list(KEYPOINT_NAMES)
    scale = torso_length(xy, names)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros((len(seq.frames), N_FEATURES))
    idx = {n: i for i, n in enumerate(names)}
    hips = midpoint(xy[:, idx["l_hip"]], xy[:, idx["r_hip"]])
    order = [idx[n] for n in role_order(seq.bowling_arm)]
    rel = (xy[:, order, :] - hips[:, None, :]) / scale
    visible = np.asarray(np.isfinite(rel).all(axis=2), dtype=bool)
    vel = np.zeros_like(rel)
    vel[1:] = (rel[1:] - rel[:-1]) * seq.fps / 10.0  # torso lengths per 0.1 s
    vel[~(visible & np.roll(visible, 1, axis=0))] = 0.0
    vel[0] = 0.0
    rel = np.where(visible[..., None], rel, 0.0)
    feats = np.concatenate([rel, vel, visible[..., None].astype(np.float64)], axis=2)
    out = feats.reshape(len(seq.frames), N_FEATURES)
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
