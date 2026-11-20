"""Module G: compare an action with reference actions (DTW) and score consistency.

Per frame, five joint angles (deg / 180, so 0-1): bowling elbow, front knee, bowling
shoulder (trunk vs upper arm), trunk lean from vertical, and shoulder-line vs hip-line tilt.
The action from back-foot contact to the end of the follow-through is resampled to a fixed
number of points (time-normalised), then compared by DTW (tslearn).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from bowlsense.analysis.legality import elbow_angles
from bowlsense.config import CompareSettings
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.angles import FloatArray, joint_angle, keypoint_track, line_tilt, midpoint
from bowlsense.geometry.body import joint
from bowlsense.render.overlay import front_knee_angles
from bowlsense.render.report_pdf import trunk_lean
from bowlsense.schemas import (
    ConsistencyResult,
    DeliveryAnalysis,
    PhaseEvents,
    PoseSequence,
    ReferenceComparison,
)

JOINTS: tuple[str, ...] = ("elbow", "front_knee", "shoulder", "trunk_lean", "hip_shoulder_tilt")


class ReferenceAction(BaseModel):
    """A reference action shipped in ``data/reference/`` (never named after a real player)."""

    name: str
    description: str
    sequence: PoseSequence  # cleaned keypoints
    phases: PhaseEvents


def angle_features(seq: PoseSequence) -> FloatArray:
    """``(frames, 5)`` joint angles in degrees (``NaN`` where not visible)."""
    arm = seq.bowling_arm
    sh = keypoint_track(seq, joint(arm, "bowling", "shoulder"))
    hips = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
    shoulder = joint_angle(hips, sh, keypoint_track(seq, joint(arm, "bowling", "elbow")))
    tilt = line_tilt(
        keypoint_track(seq, "l_shoulder"), keypoint_track(seq, "r_shoulder")
    ) - line_tilt(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
    return np.stack(
        [elbow_angles(seq), front_knee_angles(seq), shoulder, trunk_lean(seq), tilt], axis=1
    )


def action_curve(seq: PoseSequence, phases: PhaseEvents, points: int) -> FloatArray:
    """BFC -> follow-through end, resampled to ``points`` x 5, scaled to 0-1 (deg / 180).

    Gaps are linearly interpolated; a joint never seen is 0.

    Raises:
        InvalidInputError: if the window is shorter than 2 frames.
    """
    lo, hi = phases.bfc, min(phases.follow_through_end, len(seq.frames) - 1)
    if hi - lo < 1:
        raise InvalidInputError(
            "The action window (BFC to follow-through) is too short to compare."
        )
    feats = angle_features(seq)[lo : hi + 1]
    t_src = np.linspace(0.0, 1.0, len(feats))
    t_dst = np.linspace(0.0, 1.0, points)
    out = np.zeros((points, feats.shape[1]))
    for j in range(feats.shape[1]):
        ok = np.isfinite(feats[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(t_dst, t_src[ok], feats[ok, j]) / 180.0
    return out


def compare(user: FloatArray, ref: FloatArray, name: str, scale: float) -> ReferenceComparison:
    """DTW between two action curves: similarity % and per-joint deviation (deg)."""
    from tslearn.metrics import dtw_path

    path, dist = dtw_path(user, ref)
    idx_u = np.array([p[0] for p in path])
    idx_r = np.array([p[1] for p in path])
    dev = np.abs(user[idx_u] - ref[idx_r]).mean(axis=0) * 180.0
    return ReferenceComparison(
        name=name,
        similarity_pct=float(100.0 * np.exp(-float(dist) / scale)),
        deviation_deg={j: float(d) for j, d in zip(JOINTS, dev, strict=True)},
    )


def load_references(folder: Path) -> list[ReferenceAction]:
    """Every reference action JSON in ``folder`` (sorted by file name)."""
    return [
        ReferenceAction.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(folder.glob("*.json"))
    ]


def compare_to_references(
    seq: PoseSequence,
    phases: PhaseEvents,
    references: Sequence[ReferenceAction],
    settings: CompareSettings,
) -> list[ReferenceComparison]:
    """Compare one delivery with every reference, most similar first."""
    n = settings.resample_points
    user = action_curve(seq, phases, n)
    results = [
        compare(user, action_curve(r.sequence, r.phases, n), r.name, settings.similarity_scale)
        for r in references
    ]
    return sorted(results, key=lambda r: -r.similarity_pct)


def _metric(d: DeliveryAnalysis, name: str) -> float | None:
    m = d.faults.measurements
    values = {
        "technique_score": d.faults.technique_score,
        "elbow_extension_deg": d.legality.elbow_extension_deg,
        "stride_pct": m.get("OVERSTRIDE"),
        "release_height_pct": m.get("LOW_RELEASE"),
        "knee_collapse_deg": m.get("KNEE_COLLAPSE"),
    }
    if name not in values:
        raise InvalidInputError(f"Unknown consistency metric '{name}'.")
    return values[name]


def consistency(
    deliveries: Sequence[DeliveryAnalysis], settings: CompareSettings
) -> ConsistencyResult:
    """``100 - mean coefficient of variation`` of the key metrics, in %.

    The variation of each metric is its standard deviation divided by the mean absolute value
    across deliveries (metrics near zero, like elbow extension, use a 10-unit floor so tiny
    means don't explode the ratio). Needs at least 2 deliveries.
    """
    n = len(deliveries)
    variation: dict[str, float] = {}
    for name in settings.consistency_metrics:
        vals = np.array([v for d in deliveries if (v := _metric(d, name)) is not None])
        if len(vals) >= 2:
            variation[name] = float(100.0 * vals.std() / max(np.abs(vals).mean(), 10.0))
    if n < 2 or not variation:
        return ConsistencyResult(score=None, variation_pct=variation, n_deliveries=n)
    score = float(np.clip(100.0 - np.mean(list(variation.values())), 0.0, 100.0))
    return ConsistencyResult(score=score, variation_pct=variation, n_deliveries=n)
