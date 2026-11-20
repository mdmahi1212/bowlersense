"""Tests for analysis/legality.py (module A)."""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
import pytest

from bowlsense.analysis.legality import (
    assess_legality,
    elbow_angles,
    elbow_extension,
    fitted_elbow_angles,
    front_foot_no_ball,
)
from bowlsense.config import Settings, load_settings
from bowlsense.errors import CalibrationError, InvalidInputError
from bowlsense.geometry.angles import keypoint_track
from bowlsense.geometry.body import estimate_height_px, joint
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import LegalityResult, PhaseEvents, PoseSequence
from synth.skeleton_gen import SynthParams, SynthResult, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)
ICC_NOTE = "2D video screening only. The official ICC test uses 3D motion capture."


@cache
def _settings() -> Settings:
    return load_settings()


def _assess(p: SynthParams) -> tuple[LegalityResult, SynthResult]:
    s = _settings()
    r = generate(p)
    seq = clean_sequence(r.sequence, s.pose)
    raw = clean_sequence(r.sequence, s.pose, smooth=False)
    phases = detect_phases(seq, estimate_height_px(seq, s.body).height_px, s.phases, raw=raw)
    return assess_legality(seq, phases, s.legality, raw=raw), r


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("fps", [120.0, 240.0])
@pytest.mark.parametrize(("extension", "flagged"), [(8.0, False), (22.0, True)])
def test_gate_flag_and_measure_within_3deg(
    arm: str, fps: float, extension: float, flagged: bool
) -> None:
    """Section 10: 8 deg -> not flagged; 22 deg -> flagged; measured within +-3 deg."""
    res, r = _assess(SynthParams(elbow_extension_deg=extension, bowling_arm=arm, fps=fps))
    assert res.flagged is flagged
    assert res.elbow_extension_deg == pytest.approx(r.truth.elbow_extension_deg, abs=3.0)
    assert res.note.startswith(ICC_NOTE)


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("seed", range(4))
def test_flag_decision_robust_to_2px_noise(arm: str, seed: int) -> None:
    common = {"bowling_arm": arm, "noise_px": 2.0, "seed": seed, "height_px": 540.0}
    legal, _ = _assess(SynthParams.model_validate({**common, "elbow_extension_deg": 8.0}))
    illegal, _ = _assess(SynthParams.model_validate({**common, "elbow_extension_deg": 22.0}))
    assert not legal.flagged and illegal.flagged


@pytest.mark.parametrize("seed", range(3))
def test_measure_within_3deg_with_1px_noise_at_240fps(seed: int) -> None:
    res, r = _assess(
        SynthParams(elbow_extension_deg=15.0, fps=240, noise_px=1.0, seed=seed, height_px=540)
    )
    assert res.elbow_extension_deg == pytest.approx(r.truth.elbow_extension_deg, abs=3.0)


def test_straight_arm_reads_near_zero() -> None:
    res, _ = _assess(SynthParams(elbow_extension_deg=0.0))
    assert res.elbow_extension_deg < 2.0 and not res.flagged


def test_threshold_from_config() -> None:
    s = _settings()
    strict = s.legality.model_copy(update={"elbow_extension_threshold_deg": 5.0})
    r = generate(SynthParams(elbow_extension_deg=8.0))
    seq = clean_sequence(r.sequence, s.pose)
    res = assess_legality(seq, r.phases, strict)
    assert res.flagged and res.threshold_deg == 5.0


@pytest.mark.parametrize(("fps", "expected"), [(240.0, "high"), (120.0, "medium"), (100.0, "low")])
def test_confidence_levels(fps: float, expected: str) -> None:
    res, _ = _assess(SynthParams(fps=fps))
    assert res.confidence == expected


def test_low_keypoint_confidence_is_not_high() -> None:
    s = _settings()
    r = generate(SynthParams(fps=240))
    frames = [
        f.model_copy(
            update={
                "keypoints": {
                    k: (v.model_copy(update={"conf": 0.5}) if v is not None else None)
                    for k, v in f.keypoints.items()
                }
            }
        )
        for f in r.sequence.frames
    ]
    seq = clean_sequence(r.sequence.model_copy(update={"frames": frames}), s.pose)
    assert assess_legality(seq, r.phases, s.legality).confidence == "medium"


def test_front_view_is_low_confidence_with_note() -> None:
    res, _ = _assess(SynthParams(view="front", fps=240))
    assert res.confidence == "low"
    assert res.note.startswith(ICC_NOTE) and "side-on" in res.note


# --- formula unit tests ---------------------------------------------------------------------


def test_elbow_extension_hand_computed() -> None:
    angles = np.array([150.0, 170.0, 160.0, 165.0, 175.0, 172.0, 140.0])
    # window [1, 5]: max 175 at frame 4; min over [1, 4] = 160 -> 15
    assert elbow_extension(angles, 1, 5) == pytest.approx(15.0)
    # flexion after the max doesn't count
    assert elbow_extension(np.array([170.0, 175.0, 120.0]), 0, 2) == pytest.approx(5.0)
    # NaN frames are skipped
    assert elbow_extension(np.array([160.0, np.nan, 172.0]), 0, 2) == pytest.approx(12.0)


def test_elbow_extension_validation() -> None:
    with pytest.raises(InvalidInputError):
        elbow_extension(np.zeros(5), 3, 2)
    with pytest.raises(InvalidInputError):
        elbow_extension(np.zeros(5), 0, 9)
    with pytest.raises(InvalidInputError, match="not visible"):
        elbow_extension(np.full(5, np.nan), 0, 4)


def test_fitted_angles_recover_a_smooth_straightening() -> None:
    s = _settings().legality
    t = np.arange(40.0)
    true = 160.0 + 15.0 / (1.0 + np.exp(-(t - 20.0) / 2.0))
    noisy = true + np.random.default_rng(0).normal(0.0, 1.0, t.size)
    fitted = fitted_elbow_angles(noisy, 14, 26, 120.0, s)
    assert np.isnan(fitted[:14]).all() and np.isnan(fitted[27:]).all()
    measured = elbow_extension(fitted, 14, 26)
    assert measured == pytest.approx(elbow_extension(true, 14, 26), abs=1.5)


def test_fitted_angles_need_enough_frames() -> None:
    s = _settings().legality
    a = np.full(30, np.nan)
    a[10:12] = 170.0
    with pytest.raises(InvalidInputError, match="Too few"):
        fitted_elbow_angles(a, 10, 12, 120.0, s)


def test_elbow_angles_follow_bowling_arm() -> None:
    right = generate(SynthParams()).sequence
    left = generate(SynthParams(bowling_arm="left")).sequence
    np.testing.assert_allclose(elbow_angles(right), elbow_angles(left), atol=1e-6)


# --- no-ball ---------------------------------------------------------------------------------


def _foot_x(seq: PoseSequence, phases: PhaseEvents, part: str) -> tuple[float, float]:
    p = keypoint_track(seq, joint(seq.bowling_arm, "front", part))[phases.ffc]
    return float(p[0]), float(p[1])


@pytest.mark.parametrize("arm", ["right", "left"])
def test_no_ball_by_crease_position(arm: str) -> None:
    r = generate(SynthParams(bowling_arm=arm))
    seq, ph = r.sequence, r.phases
    hx, hy = _foot_x(seq, ph, "heel")
    # The run is towards +x, so the bowler's side of a vertical crease is smaller x.
    crease_ahead = [(hx + 10.0, hy - 100.0), (hx + 10.0, hy + 100.0)]  # heel behind -> legal
    crease_behind = [(hx - 10.0, hy - 100.0), (hx - 10.0, hy + 100.0)]  # heel over -> no-ball
    on_line = [(hx, hy - 100.0), (hx, hy + 100.0)]
    assert front_foot_no_ball(seq, ph, crease_ahead) is False
    assert front_foot_no_ball(seq, ph, crease_behind) is True
    assert front_foot_no_ball(seq, ph, on_line) is True
    # point order of the crease doesn't matter
    assert front_foot_no_ball(seq, ph, crease_ahead[::-1]) is False


def test_no_ball_falls_back_to_toe() -> None:
    r = generate(SynthParams())
    seq, ph = r.sequence, r.phases
    frames = list(seq.frames)
    f = frames[ph.ffc]
    frames[ph.ffc] = f.model_copy(update={"keypoints": {**f.keypoints, "l_heel": None}})
    seq = seq.model_copy(update={"frames": frames})
    tx, ty = _foot_x(seq, ph, "toe")
    # crease just behind the toe: toe is over it -> no-ball
    assert front_foot_no_ball(seq, ph, [(tx - 5, ty - 50), (tx - 5, ty + 50)]) is True
    assert front_foot_no_ball(seq, ph, [(tx + 5, ty - 50), (tx + 5, ty + 50)]) is False


def test_no_ball_validation() -> None:
    r = generate(SynthParams())
    seq, ph = r.sequence, r.phases
    with pytest.raises(CalibrationError):
        front_foot_no_ball(seq, ph, [(1.0, 1.0)])
    with pytest.raises(CalibrationError, match="identical"):
        front_foot_no_ball(seq, ph, [(1.0, 1.0), (1.0, 1.0)])
    bx, by = keypoint_track(seq, "r_ankle")[ph.bfc]
    with pytest.raises(CalibrationError, match="back foot"):
        front_foot_no_ball(seq, ph, [(bx, by - 50), (bx, by + 50)])


def test_assess_legality_includes_no_ball() -> None:
    s = _settings()
    r = generate(SynthParams())
    hx, hy = _foot_x(r.sequence, r.phases, "heel")
    res = assess_legality(
        r.sequence, r.phases, s.legality, popping_crease=[(hx + 10, hy - 50), (hx + 10, hy + 50)]
    )
    assert res.front_foot_no_ball is False
    assert assess_legality(r.sequence, r.phases, s.legality).front_foot_no_ball is None


# --- screening gate: never flag what can't be measured ------------------------------------------


def test_untrustworthy_clips_are_not_screened() -> None:
    """A clean 22 deg throw is flagged; the same clip is not flagged when the measurement
    can't be trusted (low fps, front view, broken phases, long window, implausible value)."""
    s = load_settings()
    r = generate(SynthParams(elbow_extension_deg=22.0, fps=120.0))
    seq = clean_sequence(r.sequence, s.pose)
    ok = assess_legality(seq, r.phases, s.legality)
    assert ok.flagged and ok.screened and ok.not_screened_reason is None

    def check(seq_: object, phases: object, settings: object, needle: str) -> None:
        res = assess_legality(seq_, phases, settings)  # type: ignore[arg-type]
        assert not res.flagged and not res.screened and res.confidence == "low"
        assert res.not_screened_reason is not None and needle in res.not_screened_reason
        assert res.note.startswith(ICC_NOTE) and "Not screened" in res.note

    leg = s.legality
    check(seq, r.phases, leg.model_copy(update={"screen_min_fps": 200.0}), "fps")
    check(seq.model_copy(update={"view": "front"}), r.phases, leg, "side-on")
    stride_after = r.phases.model_copy(update={"bfc": r.phases.release + 1, "confidence": 0.2})
    check(seq, stride_after, leg, "after the delivery stride")
    # an impossible arm-horizontal falls back to the last 80 ms before release (low confidence)
    early = r.phases.model_copy(update={"arm_horizontal": r.phases.bfc - 30, "confidence": 0.2})
    res = assess_legality(seq, early, leg)
    assert res.screened and res.confidence == "low" and "last 80 ms" in res.note
    assert res.verdict == "not_legal" and res.elbow_extension_deg > 15.0
    check(seq, r.phases, leg.model_copy(update={"screen_max_plausible_deg": 10.0}), "beyond")


def test_verdict_legal_not_legal_cannot_judge() -> None:
    s = load_settings()
    for ext, verdict in ((8.0, "legal"), (22.0, "not_legal")):
        r = generate(SynthParams(elbow_extension_deg=ext, fps=120.0))
        res = assess_legality(clean_sequence(r.sequence, s.pose), r.phases, s.legality)
        assert res.verdict == verdict
        word = "Legal:" if verdict == "legal" else "Not legal:"
        assert res.verdict_text.startswith(word) and "15 deg limit" in res.verdict_text
    r = generate(SynthParams(elbow_extension_deg=22.0, fps=120.0))
    seq = clean_sequence(r.sequence, s.pose).model_copy(update={"view": "front"})
    res = assess_legality(seq, r.phases, s.legality)
    assert res.verdict == "cannot_judge" and res.verdict_text.startswith("Can't judge")
    assert not res.flagged
