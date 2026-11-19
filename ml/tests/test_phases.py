"""Tests for phases/heuristic.py (spec Section 6 and the Section 10 phase tests)."""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
import pytest

from bowlsense.config import Settings, load_settings
from bowlsense.errors import InvalidInputError, PhaseDetectionError
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import (
    detect_phases,
    find_arm_horizontal,
    find_release,
    first_sustained,
    fit_impact_time,
)
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import PhaseEvents, PoseSequence
from synth.skeleton_gen import SynthParams, generate

EVENTS = (
    "run_up_start",
    "bound",
    "bfc",
    "ffc",
    "arm_horizontal",
    "release",
    "follow_through_end",
)
# A bowler filling about half of a 1080p frame: full body, side-on, ~10 m away, as the
# recording tips ask. 2 px of noise at this size is ~0.4% of body height.
REALISTIC_HEIGHT_PX = 540.0

VARIANTS: list[dict[str, object]] = [
    {},
    {"bowling_arm": "left"},
    {"stride_pct": 95, "knee_flex_change_deg": 30},
    {"action_type": "mixed", "counter_rotation_deg": 35, "bowling_arm": "left"},
    {"runup_decel_pct": 30, "follow_through_pct": 40},
    {"elbow_extension_deg": 22},
    {"release_height_pct": 125, "bowling_arm": "left"},
    {"stride_pct": 62, "action_type": "side_on"},
]


@cache
def _settings() -> Settings:
    return load_settings()


def _run(params: SynthParams, *, use_raw: bool = True) -> tuple[PhaseEvents, PhaseEvents, int]:
    s = _settings()
    r = generate(params)
    seq = clean_sequence(r.sequence, s.pose)
    raw = clean_sequence(r.sequence, s.pose, smooth=False) if use_raw else None
    height = estimate_height_px(seq, s.body).height_px
    return detect_phases(seq, height, s.phases, raw=raw), r.phases, len(seq.frames)


def _errors(found: PhaseEvents, truth: PhaseEvents) -> dict[str, int]:
    return {e: getattr(found, e) - getattr(truth, e) for e in EVENTS}


def _params(variant: dict[str, object], **kw: object) -> SynthParams:
    return SynthParams.model_validate({**variant, **kw})


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gate_within_2_frames_noise2(variant: dict[str, object], seed: int) -> None:
    """Section 10: phases within +-2 frames at 120 fps with noise_px=2 (both arms)."""
    found, truth, _ = _run(
        _params(variant, fps=120, noise_px=2.0, seed=seed, height_px=REALISTIC_HEIGHT_PX)
    )
    errs = _errors(found, truth)
    assert all(abs(e) <= 2 for e in errs.values()), errs
    assert found.confidence == 1.0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("seed", [3, 4])
def test_gate_dropout_within_3_frames(variant: dict[str, object], seed: int) -> None:
    """Section 10 robustness: 10% keypoint dropout still gives phases within +-3 frames."""
    found, truth, _ = _run(
        _params(
            variant,
            noise_px=2.0,
            dropout_rate=0.1,
            seed=seed,
            height_px=REALISTIC_HEIGHT_PX,
        )
    )
    errs = _errors(found, truth)
    assert all(abs(e) <= 3 for e in errs.values()), errs


@pytest.mark.parametrize("variant", VARIANTS[:4])
def test_clean_within_1_frame(variant: dict[str, object]) -> None:
    found, truth, _ = _run(_params(variant))
    errs = _errors(found, truth)
    assert all(abs(e) <= 1 for e in errs.values()), errs


@pytest.mark.parametrize("seed", range(6))
def test_small_figure_stress(seed: int) -> None:
    """A 300 px bowler with 2 px noise (0.67% of height): foot strikes are at the edge of
    what the data can resolve, so contacts may slip to 3 frames; everything else stays
    within 2."""
    found, truth, _ = _run(SynthParams(noise_px=2.0, seed=seed, height_px=300))
    errs = _errors(found, truth)
    assert all(abs(errs[e]) <= 2 for e in EVENTS if e not in ("bfc", "ffc")), errs
    assert abs(errs["bfc"]) <= 3 and abs(errs["ffc"]) <= 3, errs


def test_240fps_same_timing() -> None:
    found, truth, _ = _run(SynthParams(fps=240, noise_px=2.0, height_px=REALISTIC_HEIGHT_PX))
    for e, err in _errors(found, truth).items():
        assert abs(err) / 240 <= 2 / 120, (e, err)


def test_without_raw_track_is_still_close() -> None:
    found, truth, _ = _run(SynthParams(height_px=REALISTIC_HEIGHT_PX), use_raw=False)
    errs = _errors(found, truth)
    assert all(abs(e) <= 3 for e in errs.values()), errs


@pytest.mark.parametrize("arm", ["right", "left"])
def test_front_view_measurable_events(arm: str) -> None:
    found, truth, n = _run(
        SynthParams(
            view="front",
            bowling_arm=arm,
            noise_px=2.0,
            height_px=REALISTIC_HEIGHT_PX,
        )
    )
    errs = _errors(found, truth)
    for e in ("bfc", "ffc", "arm_horizontal", "release"):
        assert abs(errs[e]) <= 2, (e, errs)
    assert found.run_up_start == 0 and found.follow_through_end == n - 1
    assert found.confidence == _settings().phases.no_horizontal_motion_confidence


def test_fallback_lowers_confidence_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    s = _settings()
    r = generate(SynthParams())
    # Start the clip 4 frames before front-foot contact: no back-foot landing to find.
    cut = r.phases.ffc - 4
    frames = [
        f.model_copy(update={"frame_idx": i, "t": i / r.sequence.fps})
        for i, f in enumerate(r.sequence.frames[cut:])
    ]
    seq = clean_sequence(r.sequence.model_copy(update={"frames": frames}), s.pose)
    with caplog.at_level(logging.WARNING, logger="bowlsense.phases.heuristic"):
        found = detect_phases(seq, 300.0, s.phases)
    assert found.confidence == s.phases.low_confidence
    assert any("bfc" in rec.message or "order" in rec.message for rec in caplog.records)


def _strip(seq: PoseSequence, names: set[str]) -> PoseSequence:
    frames = [
        f.model_copy(
            update={"keypoints": {k: (None if k in names else v) for k, v in f.keypoints.items()}}
        )
        for f in seq.frames
    ]
    return seq.model_copy(update={"frames": frames})


def test_no_release_raises() -> None:
    s = _settings()
    seq = _strip(generate(SynthParams(fps=60)).sequence, {"r_wrist"})
    with pytest.raises(PhaseDetectionError, match="release"):
        detect_phases(seq, 300.0, s.phases)


def test_input_validation() -> None:
    s = _settings()
    seq = generate(SynthParams(fps=60)).sequence
    with pytest.raises(InvalidInputError):
        detect_phases(seq, 0.0, s.phases)
    with pytest.raises(InvalidInputError):
        detect_phases(seq.model_copy(update={"frames": seq.frames[:1]}), 300.0, s.phases)
    with pytest.raises(InvalidInputError, match="same number"):
        detect_phases(seq, 300.0, s.phases, raw=seq.model_copy(update={"frames": seq.frames[:10]}))


# --- unit tests of the building blocks ------------------------------------------------------


@pytest.mark.parametrize("tau", [10.0, 10.3, 12.7])
@pytest.mark.parametrize("order", [1, 2])
def test_fit_impact_time_recovers_kink(tau: float, order: int) -> None:
    t = np.arange(0.0, 25.0)
    d = np.minimum(t - tau, 0.0)
    y = 100.0 + 4.0 * d + (0.3 * d**2 if order == 2 else 0.0)
    assert fit_impact_time(t, y, 5.0, 20.0, order) == pytest.approx(tau, abs=0.11)


def test_fit_impact_time_ignores_nan_and_noise() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 30.0)
    y = 200.0 + 5.0 * np.minimum(t - 15.4, 0.0) + rng.normal(0, 0.5, t.size)
    y[[3, 20]] = np.nan
    assert fit_impact_time(t, y, 8.0, 22.0, 1) == pytest.approx(15.4, abs=0.5)


def test_first_sustained() -> None:
    m = [False, True, False, True, True, True, False]
    assert first_sustained(m, 3) == 3
    assert first_sustained(m, 1) == 1
    assert first_sustained(m, 4) is None
    assert first_sustained([False, True, True], 5, start=1) == 1  # shorter at clip end


def test_find_release_and_arm_horizontal_on_simple_arc() -> None:
    n = 61
    # Arm swings up through horizontal and over the top, fastest when pointing straight up.
    u = np.linspace(0.0, 1.0, n)
    ang = np.radians(90.0 + 150.0 * np.sin((u - 0.5) * np.pi / 1.2))
    shoulder = np.tile([100.0, 100.0], (n, 1))
    elbow = shoulder + 30 * np.stack([np.cos(ang), -np.sin(ang)], axis=1)
    wrist = shoulder + 60 * np.stack([np.cos(ang), -np.sin(ang)], axis=1)
    release = find_release(wrist, shoulder, fps=100.0, refine_frames=3)
    # wrist height peaks when the arm points straight up (90 deg)
    assert abs(np.degrees(ang[release]) - 90.0) < 360.0 / n
    ah = find_arm_horizontal(shoulder, elbow, release)
    assert ah is not None and np.degrees(ang[ah - 1]) < 0.0 <= np.degrees(ang[ah])
