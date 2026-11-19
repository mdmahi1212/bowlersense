"""Tests for analysis/faults.py (module C)."""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import pytest
import yaml

from bowlsense.analysis.faults import FAULT_SPECS, assess_faults, severity
from bowlsense.config import DEFAULT_CONFIG_PATH, FAULT_CODES, Settings, load_settings
from bowlsense.errors import ConfigError, InvalidInputError
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import FaultReport
from synth.skeleton_gen import SynthParams, SynthResult, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)
SPEC = {spec.code: spec for spec in FAULT_SPECS}


@cache
def _settings() -> Settings:
    return load_settings()


@cache
def _report(**kw: object) -> tuple[FaultReport, SynthResult]:
    s = _settings()
    r = generate(SynthParams.model_validate({"height_px": 540.0, **kw}))
    seq = clean_sequence(r.sequence, s.pose)
    raw = clean_sequence(r.sequence, s.pose, smooth=False)
    height = estimate_height_px(seq, s.body).height_px
    phases = detect_phases(seq, height, s.phases, raw=raw)
    return assess_faults(seq, phases, height, s, raw=raw), r


def _fired(report: FaultReport) -> set[str]:
    return {f.code for f in report.faults}


# (fault code, params below the threshold, params past the threshold)
FIRE_CASES: list[tuple[str, dict[str, object], dict[str, object]]] = [
    ("KNEE_COLLAPSE", {"knee_flex_change_deg": 8.0}, {"knee_flex_change_deg": 32.0}),
    ("OVERSTRIDE", {"stride_pct": 80.0}, {"stride_pct": 98.0}),
    ("UNDERSTRIDE", {"stride_pct": 75.0}, {"stride_pct": 60.0}),
    ("RUNUP_DECEL", {"runup_decel_pct": 5.0}, {"runup_decel_pct": 28.0}),
    ("LOW_RELEASE", {"release_height_pct": 122.0}, {"release_height_pct": 111.0}),
    ("SHORT_FOLLOW", {"follow_through_pct": 80.0}, {"follow_through_pct": 20.0}),
    ("WEAK_FRONT_ARM", {"front_arm_ffc_deg": 70.0}, {"front_arm_ffc_deg": -40.0}),
    ("CHAIN_ORDER", {"pelvis_rotation_delay_s": 0.0}, {"pelvis_rotation_delay_s": 0.12}),
    (
        "HEAD_FALL",
        {"view": "front", "lateral_flexion_deg": 10.0},
        {"view": "front", "lateral_flexion_deg": 42.0},
    ),
    (
        "LATERAL_FLEXION",
        {"view": "front", "lateral_flexion_deg": 15.0},
        {"view": "front", "lateral_flexion_deg": 42.0},
    ),
]


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize(("code", "below", "past"), FIRE_CASES, ids=[c[0] for c in FIRE_CASES])
def test_gate_each_fault_fires_only_past_threshold(
    code: str, below: dict[str, object], past: dict[str, object], arm: str
) -> None:
    """Section 10: each fault fires when its parameter is past the threshold and doesn't
    fire when it's below (left and right arm)."""
    quiet, _ = _report(**below, bowling_arm=arm)
    loud, _ = _report(**past, bowling_arm=arm)
    assert code in quiet.measurements and code not in _fired(quiet)
    assert code in _fired(loud)
    fault = next(f for f in loud.faults if f.code == code)
    assert 0.0 <= fault.severity <= 100.0
    assert fault.explanation and fault.frames
    assert fault.threshold == SPEC[code].threshold(_settings().faults)


@pytest.mark.parametrize("seed", range(3))
def test_fire_decisions_hold_with_noise(seed: int) -> None:
    quiet, _ = _report(noise_px=2.0, seed=seed, knee_flex_change_deg=8.0)
    loud, _ = _report(noise_px=2.0, seed=seed, knee_flex_change_deg=32.0, stride_pct=98.0)
    assert not _fired(quiet) & {"KNEE_COLLAPSE", "OVERSTRIDE"}
    assert {"KNEE_COLLAPSE", "OVERSTRIDE"} <= _fired(loud)


def test_measurements_match_truth_side_view() -> None:
    rep, r = _report(knee_flex_change_deg=25.0, runup_decel_pct=20.0, stride_pct=88.0)
    m, t = rep.measurements, r.truth
    assert m["KNEE_COLLAPSE"] == pytest.approx(t.knee_flex_change_deg, abs=3.0)
    assert m["OVERSTRIDE"] == pytest.approx(t.stride_pct, abs=1.5)
    assert m["UNDERSTRIDE"] == m["OVERSTRIDE"]
    assert m["RUNUP_DECEL"] == pytest.approx(t.runup_decel_pct, abs=3.5)
    assert m["LOW_RELEASE"] == pytest.approx(t.release_height_pct, abs=2.5)
    assert m["WEAK_FRONT_ARM"] == pytest.approx(-t.front_wrist_above_shoulder_ffc_pct, abs=1.5)
    assert m["CHAIN_ORDER"] == 0.0


def test_measurements_match_truth_front_view() -> None:
    rep, r = _report(view="front", lateral_flexion_deg=35.0)
    assert rep.measurements["HEAD_FALL"] == pytest.approx(r.truth.head_offset_pct, abs=2.0)
    assert rep.measurements["LATERAL_FLEXION"] == pytest.approx(
        r.truth.trunk_tilt_front_deg, abs=3.0
    )


def test_view_routing() -> None:
    side, _ = _report()
    front, _ = _report(view="front")
    assert set(side.measurements) | set(side.not_measured) == set(FAULT_CODES)
    assert side.not_measured == {
        "HEAD_FALL": "needs front view",
        "LATERAL_FLEXION": "needs front view",
    }
    assert set(front.measurements) == {"HEAD_FALL", "LATERAL_FLEXION"}
    assert all(v == "needs side view" for v in front.not_measured.values())


def test_clean_action_scores_100() -> None:
    rep, _ = _report()
    assert rep.faults == [] and rep.technique_score == 100.0


def test_technique_score_is_weighted_mean_severity() -> None:
    rep, _ = _report(knee_flex_change_deg=32.0, stride_pct=98.0, runup_decel_pct=28.0)
    w = _settings().faults.weights
    total = sum(w[c] for c in rep.measurements)
    expected = 100.0 - sum(w[f.code] * f.severity for f in rep.faults) / total
    assert rep.technique_score == pytest.approx(expected)
    assert rep.technique_score < 100.0


def test_missing_keypoints_go_to_not_measured() -> None:
    s = _settings()
    r = generate(SynthParams())
    frames = [
        f.model_copy(update={"keypoints": {**f.keypoints, "l_wrist": None, "l_shoulder": None}})
        for f in r.sequence.frames
    ]
    seq = r.sequence.model_copy(update={"frames": frames})
    rep = assess_faults(seq, r.phases, 300.0, s)
    assert "WEAK_FRONT_ARM" in rep.not_measured
    assert "not visible" in rep.not_measured["WEAK_FRONT_ARM"]


def test_invalid_height() -> None:
    r = generate(SynthParams())
    with pytest.raises(InvalidInputError):
        assess_faults(r.sequence, r.phases, 0.0, _settings())


# --- severity ramps ----------------------------------------------------------------------------


def test_severity_above_ramps_to_twice_threshold() -> None:
    fs = _settings().faults
    knee = SPEC["KNEE_COLLAPSE"]  # threshold 20, full at 40
    assert severity(knee, 20.0, fs) is None
    assert severity(knee, 30.0, fs) == pytest.approx(50.0)
    assert severity(knee, 40.0, fs) == pytest.approx(100.0)
    assert severity(knee, 90.0, fs) == pytest.approx(100.0)


def test_severity_below_uses_full_point() -> None:
    fs = _settings().faults
    under = SPEC["UNDERSTRIDE"]  # threshold 65, full at 50
    assert severity(under, 70.0, fs) is None
    assert severity(under, 57.5, fs) == pytest.approx(50.0)
    assert severity(under, 40.0, fs) == pytest.approx(100.0)
    low = SPEC["LOW_RELEASE"]  # threshold 115, full at 100
    assert severity(low, 107.5, fs) == pytest.approx(50.0)


def test_severity_zero_threshold_fault() -> None:
    fs = _settings().faults
    arm = SPEC["WEAK_FRONT_ARM"]  # threshold 0, full at 10% of height below the shoulder
    assert severity(arm, -5.0, fs) is None
    assert severity(arm, 5.0, fs) == pytest.approx(50.0)


def _raw_config() -> dict:  # type: ignore[type-arg]
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"BOGUS": 1}, "unknown codes"),
        ({"UNDERSTRIDE": 70}, "UNDERSTRIDE"),
        ({"LOW_RELEASE": 120}, "LOW_RELEASE"),
        ({"WEAK_FRONT_ARM": 0}, "WEAK_FRONT_ARM"),
    ],
)
def test_severity_full_config_validation(
    tmp_path: Path, update: dict[str, float], match: str
) -> None:
    raw = _raw_config()
    raw["faults"]["severity_full"] = {**raw["faults"]["severity_full"], **update}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_settings(path)


def test_severity_full_must_cover_low_type_faults(tmp_path: Path) -> None:
    raw = _raw_config()
    del raw["faults"]["severity_full"]["SHORT_FOLLOW"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="SHORT_FOLLOW"):
        load_settings(path)
