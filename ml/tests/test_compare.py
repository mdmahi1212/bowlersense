"""Tests for module G (Phase 10): DTW comparison, references, consistency, bowler ID."""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
import pytest

from bowlsense.analysis.bowler_id import BowlerIdentifier, data_status
from bowlsense.analysis.compare import (
    JOINTS,
    action_curve,
    compare,
    compare_to_references,
    consistency,
    load_references,
)
from bowlsense.config import Settings, load_settings
from bowlsense.errors import InvalidInputError
from bowlsense.pipeline import analyze_delivery
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import DeliveryAnalysis, PhaseEvents, PoseSequence
from synth.skeleton_gen import SynthParams, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


@cache
def _action(**kw: object) -> tuple[PoseSequence, PhaseEvents]:
    r = generate(SynthParams.model_validate(kw))
    return clean_sequence(r.sequence, _settings().pose), r.phases


def _curve(**kw: object) -> np.ndarray:
    seq, ph = _action(**kw)
    return action_curve(seq, ph, _settings().compare.resample_points)


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    return compare(a, b, "x", _settings().compare.similarity_scale).similarity_pct


def test_gate_identical_sequences_are_100pct() -> None:
    """Phase 10 gate: identical sequences -> ~100% similarity."""
    c = _curve(noise_px=2.0, seed=4)
    result = compare(c, c, "self", _settings().compare.similarity_scale)
    assert result.similarity_pct == pytest.approx(100.0, abs=1e-6)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in result.deviation_deg.values())
    assert set(result.deviation_deg) == set(JOINTS)


def test_similarity_falls_with_distortion() -> None:
    base = _curve()
    near = _curve(noise_px=2.0, seed=1)
    far = _curve(elbow_extension_deg=25.0, knee_flex_change_deg=35.0, lateral_flexion_deg=40.0)
    assert 100.0 > _sim(base, near) > _sim(base, far)
    dev = compare(base, far, "far", _settings().compare.similarity_scale).deviation_deg
    assert dev["front_knee"] > 5.0 and dev["elbow"] > 3.0


def test_invariant_to_fps_and_bowling_arm() -> None:
    base = _curve()
    assert _sim(base, _curve(fps=240.0)) > 90.0  # time-normalised (phases shift a frame or two)
    assert _sim(base, _curve(bowling_arm="left")) > 99.0  # joints chosen by role
    assert _sim(base, _curve(height_px=500.0)) > 99.0  # angles are scale-free


def test_shipped_references() -> None:
    refs = load_references(_settings().compare.reference_path)
    assert len(refs) >= 2
    assert all(r.name.startswith("Reference: ") for r in refs)  # never a real player
    seq, ph = _action(action_type="front_on", noise_px=2.0)
    results = compare_to_references(seq, ph, refs, _settings().compare)
    sims = [r.similarity_pct for r in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0].name == "Reference: textbook front-on"
    assert all(0.0 < s <= 100.0 for s in sims)


def test_action_curve_edge_cases() -> None:
    seq, ph = _action()
    with pytest.raises(InvalidInputError, match="too short"):
        action_curve(seq, ph.model_copy(update={"follow_through_end": ph.bfc}), 50)
    frames = [
        f.model_copy(update={"keypoints": {**f.keypoints, "l_knee": None}}) for f in seq.frames
    ]
    blind = action_curve(seq.model_copy(update={"frames": frames}), ph, 50)
    assert np.all(blind[:, JOINTS.index("front_knee")] == 0.0)
    assert blind.shape == (50, len(JOINTS)) and np.all(np.abs(blind) <= 1)  # tilt is signed


# --- consistency ---------------------------------------------------------------------------------


@cache
def _delivery(**kw: object) -> DeliveryAnalysis:
    return analyze_delivery(generate(SynthParams.model_validate(kw)).sequence, _settings())


def test_consistency_identical_vs_varied() -> None:
    s = _settings().compare
    same = consistency([_delivery(), _delivery()], s)
    assert same.score == pytest.approx(100.0) and same.n_deliveries == 2
    deliveries = [
        _delivery(),
        _delivery(stride_pct=95.0, knee_flex_change_deg=30.0),
        _delivery(release_height_pct=110.0, elbow_extension_deg=20.0),
    ]
    varied = consistency(deliveries, s)
    assert varied.score is not None and varied.score < 95.0
    assert varied.variation_pct["knee_collapse_deg"] > 20.0  # 8 deg vs 30 deg collapse


def test_consistency_needs_two_deliveries() -> None:
    r = consistency([_delivery()], _settings().compare)
    assert r.score is None and r.n_deliveries == 1


def test_consistency_unknown_metric() -> None:
    s = _settings().compare.model_copy(update={"consistency_metrics": ("nope",)})
    with pytest.raises(InvalidInputError, match="Unknown consistency metric"):
        consistency([_delivery(), _delivery()], s)


# --- bowler ID ------------------------------------------------------------------------------------

STYLES: dict[str, dict[str, object]] = {
    "bowler_a": {"action_type": "side_on", "stride_pct": 70.0, "elbow_extension_deg": 3.0},
    "bowler_b": {"action_type": "front_on", "stride_pct": 90.0, "knee_flex_change_deg": 28.0},
    "bowler_c": {"action_type": "mixed", "lateral_flexion_deg": 35.0, "elbow_extension_deg": 12.0},
}


def _bowler_data(n: int, seed0: int) -> tuple[list[np.ndarray], list[str]]:
    curves, labels = [], []
    for name, style in STYLES.items():
        for k in range(n):
            rng = np.random.default_rng(seed0 + k)
            jitter = {"release_height_pct": float(118 + rng.uniform(-3, 3)),
                      "noise_px": 1.5, "seed": seed0 + k}  # fmt: skip
            curves.append(_curve(**{**style, **jitter}))
            labels.append(name)
    return curves, labels


def test_bowler_id_insufficient_data_path() -> None:
    s = _settings().bowler_id
    ok, detail = data_status(["a"] * 25 + ["b"] * 25 + ["c"] * 5, s)
    assert not ok and "have 2" in detail
    ident = BowlerIdentifier(s)
    curves, labels = _bowler_data(4, 100)  # 3 bowlers but only 4 deliveries each
    assert ident.fit(curves, labels) is False
    result = ident.predict(curves[0])
    assert result.status == "insufficient data" and result.bowler_id is None
    assert "needs 3 bowlers" in result.detail


def test_bowler_id_trains_and_identifies() -> None:
    s = _settings().bowler_id
    train_c, train_l = _bowler_data(s.min_deliveries_per_bowler, 0)
    test_c, test_l = _bowler_data(5, 1000)
    ident = BowlerIdentifier(s)
    assert ident.fit(train_c, train_l) is True
    preds = [ident.predict(c) for c in test_c]
    assert all(p.status == "ok" and p.probability is not None for p in preds)
    accuracy = np.mean([p.bowler_id == t for p, t in zip(preds, test_l, strict=True)])
    assert accuracy >= 0.8
