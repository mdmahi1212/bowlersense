"""Delivery type (pace vs spin) from pose features around release (spec module F, stretch).

A logistic regression on scale-free features the pipeline measures itself. The trained
model is stored as plain JSON (feature names, standardisation, coefficients), so loading it
runs no code and needs neither scikit-learn nor pickle.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from bowlsense.errors import ModelNotFoundError
from bowlsense.geometry.angles import (
    FloatArray,
    keypoint_array,
    keypoint_track,
    linear_speed,
    midpoint,
    segment_angle,
)
from bowlsense.geometry.body import joint
from bowlsense.phases.features import torso_length
from bowlsense.pose.smoothing import savgol_derivative
from bowlsense.schemas import KEYPOINT_NAMES, DeliveryTypeResult, PoseSequence

FEATURES: tuple[str, ...] = (
    "wrist_speed_peak",  # torso lengths / s, bowling wrist, release -0.3 s .. +0.15 s
    "arm_angular_velocity_peak",  # deg / s, bowling upper arm, same window
    "approach_speed",  # torso lengths / s, hip midpoint, last 1 s before release
    "scale_change_rate",  # |d log(torso length) / dt|, last 1 s (running towards/away)
    "front_wrist_speed_peak",  # torso lengths / s, non-bowling wrist, same window as above
    "ankle_spread_at_release",  # torso lengths, horizontal ankle-to-ankle at release
)


def _window(n: int, centre: int, before: int, after: int) -> slice:
    return slice(max(0, centre - before), min(n, centre + after + 1))


def delivery_features(seq: PoseSequence, release: int) -> FloatArray:
    """The ``FEATURES`` vector for a cleaned sequence with the bowling arm set. ``NaN`` where
    something isn't visible (the model fills those with the training mean)."""
    fps, n = seq.fps, len(seq.frames)
    xy, _ = keypoint_array(seq)
    scale = torso_length(xy, list(KEYPOINT_NAMES))
    out = np.full(len(FEATURES), np.nan)
    if not np.isfinite(scale) or scale <= 0 or not 0 <= release < n:
        return out
    arm = seq.bowling_arm
    near = _window(n, release, round(0.3 * fps), round(0.15 * fps))
    last_s = _window(n, release, round(1.0 * fps), 0)

    def peak(values: FloatArray, sl: slice) -> float:
        seg = values[sl]
        return float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan

    wrist = keypoint_track(seq, joint(arm, "bowling", "wrist"))
    out[0] = peak(linear_speed(wrist, fps), near) / scale
    upper = segment_angle(
        keypoint_track(seq, joint(arm, "bowling", "shoulder")),
        keypoint_track(seq, joint(arm, "bowling", "elbow")),
    )
    ok = np.isfinite(upper)
    if ok.sum() > 3:
        unwrapped = np.full(n, np.nan)
        unwrapped[ok] = np.degrees(np.unwrap(np.radians(upper[ok])))
        out[1] = peak(np.abs(savgol_derivative(unwrapped, fps, 0.1)), near)
    hips = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
    seg = hips[last_s]
    good = np.isfinite(seg).all(axis=1)
    if good.sum() >= 2:
        pts = seg[good]
        out[2] = float(np.linalg.norm(pts[-1] - pts[0])) / scale / (len(seg) / fps)
    sh = midpoint(keypoint_track(seq, "l_shoulder"), keypoint_track(seq, "r_shoulder"))
    torso = np.linalg.norm(sh - hips, axis=1)[last_s]
    t_ok = np.isfinite(torso) & (torso > 0)
    if t_ok.sum() >= 4:
        t = np.arange(len(torso))[t_ok] / fps
        out[3] = abs(float(np.polyfit(t, np.log(torso[t_ok]), 1)[0]))
    front_wrist = keypoint_track(seq, joint(arm, "non_bowling", "wrist"))
    out[4] = peak(linear_speed(front_wrist, fps), near) / scale
    la = keypoint_track(seq, "l_ankle")[release]
    ra = keypoint_track(seq, "r_ankle")[release]
    if np.isfinite(la).all() and np.isfinite(ra).all():
        out[5] = abs(float(la[0] - ra[0])) / scale
    return out


class DeliveryTypeModel(BaseModel):
    """Standardised logistic regression, stored as JSON."""

    feature_names: list[str]
    means: list[float]
    scales: list[float]
    coef: list[float]
    intercept: float
    threshold: float = 0.5
    metrics: dict[str, float] = {}

    def p_fast(self, features: FloatArray) -> float:
        """Probability that the delivery is pace (fast)."""
        if list(FEATURES) != self.feature_names:
            raise ValueError("delivery-type model was trained on different features; retrain")
        x = np.where(np.isfinite(features), features, np.asarray(self.means))
        z = (x - np.asarray(self.means)) / np.asarray(self.scales)
        logit = float(z @ np.asarray(self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def classify(self, seq: PoseSequence, release: int) -> DeliveryTypeResult:
        """Pace or spin for one delivery."""
        p = self.p_fast(delivery_features(seq, release))
        return DeliveryTypeResult(delivery_type="fast" if p >= self.threshold else "spin", p_fast=p)


@lru_cache(maxsize=2)
def load_delivery_type_model(path: Path) -> DeliveryTypeModel:
    """Load the JSON model from ``models/``.

    Raises:
        ModelNotFoundError: if it hasn't been trained yet.
    """
    if not path.is_file():
        raise ModelNotFoundError(
            path.name, path.parent, "Train it with `python -m train.train_delivery_type`."
        )
    return DeliveryTypeModel.model_validate_json(path.read_text(encoding="utf-8"))
