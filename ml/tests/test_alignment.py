"""Tests for analysis/alignment.py (module B)."""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
import pytest

from bowlsense.analysis.alignment import (
    NEEDS_FRONT_VIEW,
    assess_alignment,
    classify_action,
    fitted_minimum,
)
from bowlsense.config import Settings, load_settings
from bowlsense.errors import PoseQualityError
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import AlignmentResult
from synth.skeleton_gen import SynthParams, SynthResult, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


def _assess(p: SynthParams) -> tuple[AlignmentResult, SynthResult]:
    s = _settings()
    r = generate(p)
    seq = clean_sequence(r.sequence, s.pose)
    raw = clean_sequence(r.sequence, s.pose, smooth=False)
    phases = detect_phases(seq, estimate_height_px(seq, s.body).height_px, s.phases, raw=raw)
    return assess_alignment(seq, phases, s.alignment, raw=raw), r


ACTIONS = ("side_on", "semi_open", "front_on", "mixed")


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("counter_rotation", [0.0, 20.0, 45.0])
@pytest.mark.parametrize(("view", "arm"), [("front", "right"), ("behind", "left")])
def test_gate_counter_rotation_within_5deg(
    action: str, counter_rotation: float, view: str, arm: str
) -> None:
    """Section 10: counter-rotation recovered within +-5 deg (left and right arm)."""
    res, r = _assess(
        SynthParams.model_validate(
            {
                "action_type": action,
                "counter_rotation_deg": counter_rotation,
                "view": view,
                "bowling_arm": arm,
                "height_px": 540.0,
            }
        )
    )
    assert res.counter_rotation_deg is not None
    assert res.counter_rotation_deg == pytest.approx(r.truth.counter_rotation_deg, abs=5.0)
    assert res.action_type == action


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("view", ["front", "behind"])
@pytest.mark.parametrize("seed", range(3))
def test_gate_mixed_classified_as_mixed_with_noise(arm: str, view: str, seed: int) -> None:
    """Section 10: a mixed action is classified as mixed (here with 2 px noise)."""
    res, _ = _assess(
        SynthParams.model_validate(
            {
                "action_type": "mixed",
                "view": view,
                "bowling_arm": arm,
                "noise_px": 2.0,
                "seed": seed,
                "height_px": 540.0,
            }
        )
    )
    assert res.action_type == "mixed"
    assert res.hip_shoulder_diff_bfc_deg is not None
    assert abs(res.hip_shoulder_diff_bfc_deg) > _settings().alignment.mixed_threshold_deg


@pytest.mark.parametrize("action", ["side_on", "semi_open", "front_on"])
@pytest.mark.parametrize("seed", range(2))
def test_non_mixed_actions_with_noise(action: str, seed: int) -> None:
    res, _ = _assess(
        SynthParams.model_validate(
            {
                "action_type": action,
                "view": "front",
                "noise_px": 2.0,
                "seed": seed,
                "height_px": 540.0,
            }
        )
    )
    assert res.action_type == action


def test_angles_at_bfc_match_truth() -> None:
    res, r = _assess(SynthParams(action_type="mixed", view="front", height_px=540))
    assert res.shoulder_angle_bfc_deg == pytest.approx(r.truth.shoulder_yaw_bfc_deg, abs=3.0)
    assert res.hip_angle_bfc_deg == pytest.approx(r.truth.hip_yaw_bfc_deg, abs=3.0)


@pytest.mark.parametrize(("counter_rotation", "flagged"), [(15.0, False), (45.0, True)])
def test_counter_rotation_flag(counter_rotation: float, flagged: bool) -> None:
    res, _ = _assess(
        SynthParams(
            action_type="side_on",
            counter_rotation_deg=counter_rotation,
            view="front",
            height_px=540,
        )
    )
    assert res.counter_rotation_flagged is flagged


def test_side_view_reports_needs_front_view() -> None:
    res, _ = _assess(SynthParams(view="side"))
    assert res.action_type == "unknown"
    assert res.counter_rotation_deg is None and res.shoulder_angle_bfc_deg is None
    assert res.counter_rotation_flagged is None and res.hip_shoulder_sep_ffc_deg is None
    assert any(NEEDS_FRONT_VIEW in n for n in res.notes)


def test_works_without_raw_track() -> None:
    s = _settings()
    r = generate(SynthParams(view="front", counter_rotation_deg=30, height_px=540))
    seq = clean_sequence(r.sequence, s.pose)
    res = assess_alignment(seq, r.phases, s.alignment)
    assert res.counter_rotation_deg is not None
    # smoothed track fills in the sharp dip: an underestimate, but in the right range
    assert 20.0 < res.counter_rotation_deg <= r.truth.counter_rotation_deg + 1.0


def test_missing_shoulders_raise() -> None:
    s = _settings()
    r = generate(SynthParams(view="front"))
    frames = [
        f.model_copy(
            update={
                "keypoints": {
                    k: (None if k.endswith("shoulder") else v) for k, v in f.keypoints.items()
                }
            }
        )
        for f in r.sequence.frames
    ]
    seq = r.sequence.model_copy(update={"frames": frames})
    with pytest.raises(PoseQualityError, match="shoulders"):
        assess_alignment(seq, r.phases, s.alignment)


@pytest.mark.parametrize(
    ("shoulder", "hip", "expected"),
    [
        (10.0, 15.0, "side_on"),
        (40.0, 45.0, "semi_open"),
        (65.0, 70.0, "front_on"),
        (15.0, 65.0, "mixed"),
        (70.0, 30.0, "mixed"),
        (25.0, 25.0, "side_on"),
        (26.0, 20.0, "semi_open"),
        (55.0, 60.0, "front_on"),
    ],
)
def test_classify_action_hand_values(shoulder: float, hip: float, expected: str) -> None:
    assert classify_action(shoulder, hip, _settings().alignment) == expected


def test_fitted_minimum_on_parabola_and_noise() -> None:
    t = np.arange(30.0)
    parabola = 0.5 * (t - 12.4) ** 2 - 20.0
    assert fitted_minimum(parabola, 0, 29, 4) == pytest.approx(-20.0, abs=1e-6)
    noisy = parabola + np.random.default_rng(1).normal(0, 1.0, t.size)
    assert fitted_minimum(noisy, 0, 29, 4) == pytest.approx(-20.0, abs=1.5)


def test_fitted_minimum_fallbacks() -> None:
    falling = np.arange(10.0, 0.0, -1.0)  # minimum at the window edge: plain min
    assert fitted_minimum(falling, 0, 9, 3) == pytest.approx(1.0)
    assert np.isnan(fitted_minimum(np.full(5, np.nan), 0, 4, 2))
    sparse = np.array([5.0, np.nan, 1.0, np.nan, 5.0])
    assert fitted_minimum(sparse, 0, 4, 2) == pytest.approx(1.0)


def test_full_width_falls_back_when_runup_window_is_empty() -> None:
    """Front-on clips can get bound <= run_up_start; calibration then uses earlier frames."""
    from bowlsense.analysis.alignment import full_width

    s = load_settings()
    r = generate(SynthParams(view="front", fps=120.0))
    seq = clean_sequence(r.sequence, s.pose)
    good = full_width(seq, r.phases, "shoulder", s.alignment.width_calib_top_frac)
    broken = r.phases.model_copy(update={"bound": r.phases.run_up_start})
    fallback = full_width(seq, broken, "shoulder", s.alignment.width_calib_top_frac)
    assert fallback == pytest.approx(good, rel=0.1)
