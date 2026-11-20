"""Learned bowler selection: which tracked person is the bowler.

Each person track gets a small feature vector (how and how much it moves, its size and size
change, where it is in the frame, how long it is seen, and the same measures relative to the
other people in the clip). A logistic regression trained on labelled match footage
(``train/train_bowler_select.py``) scores every track; the highest score is the bowler.

The model is plain JSON in ``models/`` (feature names, standardisation and weights), so it is
transparent and safe to load. Without it, ``select_bowler`` uses the hand-written rule.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.detect.person import PersonTrack
from bowlsense.errors import ModelNotFoundError
from bowlsense.geometry.angles import FloatArray

FEATURE_VERSION = 1
BASE_FEATURES: tuple[str, ...] = (
    "h_displacement",  # horizontal travel / frame width
    "h_speed",  # 95th-percentile horizontal speed / frame width (per frame)
    "extent",  # 2D travel (5-95th pct range) / own median height
    "speed_2d",  # 95th-percentile 2D speed / own median height (per frame)
    "size_change",  # log(95th / 5th pct box height)
    "coverage",  # share of the clip's frames the track is seen in
    "height",  # median box height / frame height
    "off_centre",  # |mean centre x - frame centre| / frame width
    "centre_y",  # mean centre y / frame height
)
RELATIVE = ("h_displacement", "h_speed", "extent", "speed_2d", "size_change", "height")
FEATURES: tuple[str, ...] = BASE_FEATURES + tuple(f"{n}_rel" for n in RELATIVE)


def _base(track: PersonTrack, width: int, height: int, n_frames: int) -> dict[str, float]:
    b = np.asarray(track.boxes, dtype=np.float64)
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    hm = float(np.median(h))
    cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
    frames = np.asarray(track.frames, dtype=np.float64)
    if len(frames) > 1:
        dt = np.maximum(np.diff(frames), 1.0)
        h_speed = float(np.percentile(np.abs(np.diff(cx)) / dt, 95)) / width
        speed_2d = float(np.percentile(np.hypot(np.diff(cx), np.diff(cy)) / dt, 95)) / hm
    else:
        h_speed = speed_2d = 0.0
    span = np.hypot(np.ptp(np.percentile(cx, [5, 95])), np.ptp(np.percentile(cy, [5, 95])))
    return {
        "h_displacement": float(np.ptp(cx)) / width,
        "h_speed": h_speed,
        "extent": float(span) / hm,
        "speed_2d": speed_2d,
        "size_change": float(np.log(np.percentile(h, 95) / np.percentile(h, 5))),
        "coverage": len(track.frames) / max(n_frames, 1),
        "height": hm / height,
        "off_centre": abs(float(np.mean(cx)) - width / 2.0) / width,
        "centre_y": float(np.mean(cy)) / height,
    }


def feature_matrix(tracks: list[PersonTrack], width: int, height: int, n_frames: int) -> FloatArray:
    """(tracks, FEATURES) matrix; relative features divide by the clip's maximum."""
    if not tracks:
        return np.zeros((0, len(FEATURES)))
    base = [_base(t, width, height, n_frames) for t in tracks]
    rows = []
    for f in base:
        row = [f[n] for n in BASE_FEATURES]
        for n in RELATIVE:
            top = max(x[n] for x in base)
            row.append(f[n] / top if top > 0 else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class BowlerSelector:
    """A trained logistic-regression track scorer."""

    def __init__(self, meta: dict[str, Any]) -> None:
        """Load from the JSON model; the feature list must match this code."""
        if meta.get("feature_version") != FEATURE_VERSION or tuple(meta["features"]) != FEATURES:
            raise ValueError("bowler_selector.json was trained on other features; retrain it.")
        self.mean = np.asarray(meta["mean"], dtype=np.float64)
        self.scale = np.asarray(meta["scale"], dtype=np.float64)
        self.coef = np.asarray(meta["coef"], dtype=np.float64)
        self.intercept = float(meta["intercept"])

    def scores(
        self, tracks: list[PersonTrack], width: int, height: int, n_frames: int
    ) -> FloatArray:
        """Bowler probability per track."""
        x = (feature_matrix(tracks, width, height, n_frames) - self.mean) / self.scale
        z = x @ self.coef + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


@lru_cache(maxsize=2)
def load_selector(path: Path) -> BowlerSelector:
    """Load (and cache) the trained selector.

    Raises:
        ModelNotFoundError: if the JSON file isn't in ``models/``.
    """
    if not path.is_file():
        raise ModelNotFoundError(
            path.name, path.parent, "Train it with `python -m train.train_bowler_select`."
        )
    return BowlerSelector(json.loads(path.read_text(encoding="utf-8")))
