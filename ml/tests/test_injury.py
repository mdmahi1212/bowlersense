"""Tests for analysis/injury.py (module D): monotonicity gate, scoring, synthetic end-to-end."""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import yaml

from bowlsense.analysis.alignment import assess_alignment
from bowlsense.analysis.faults import assess_faults
from bowlsense.analysis.injury import (
    COMPONENT_DRIVERS,
    arm_slot_deg,
    assess_injury,
    band_for,
    driver_score,
    injury_features,
)
from bowlsense.config import DEFAULT_CONFIG_PATH, INJURY_DRIVERS, Settings, load_settings
from bowlsense.errors import ConfigError
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import InjuryFeatures, InjuryRisk
from synth.skeleton_gen import SynthParams, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


def _component(risk: InjuryRisk, name: str) -> float:
    return next(c.score for c in risk.components if c.name == name)


def _random_features(rng: np.random.Generator) -> dict[str, object]:
    s = _settings().injury
    out: dict[str, object] = {}
    for name in INJURY_DRIVERS:
        if rng.random() < 0.25:
            continue  # some drivers missing, like a single-view clip
        low, high = s.drivers[name]
        span = high - low
        out[name] = float(rng.uniform(low - 0.5 * span, high + 0.5 * span))
    if rng.random() < 0.75:
        out["mixed_action"] = bool(rng.random() < 0.5)
    return out


NUMERIC_DRIVERS = [d for d in INJURY_DRIVERS]


@pytest.mark.parametrize("driver", NUMERIC_DRIVERS)
def test_gate_raising_a_driver_never_lowers_its_component(driver: str) -> None:
    """Phase 6 gate: monotonicity, over many random contexts and a sweep of each driver."""
    s = _settings().injury
    rng = np.random.default_rng(hash(driver) % 2**32)
    low, high = s.drivers[driver]
    sweep = np.linspace(low - (high - low), high + (high - low), 41)
    components = [c for c, drivers in COMPONENT_DRIVERS.items() if driver in drivers]
    for _ in range(40):
        context = _random_features(rng)
        previous: dict[str, float] = {}
        prev_overall = -1.0
        for value in sweep:
            risk = assess_injury(InjuryFeatures.model_validate({**context, driver: value}), s)
            for comp in components:
                score = _component(risk, comp)
                assert score >= previous.get(comp, -1.0) - 1e-9, (driver, comp, value)
                previous[comp] = score
            assert risk.overall >= prev_overall - 1e-9
            prev_overall = risk.overall


def test_gate_mixed_action_never_lowers_lumbar() -> None:
    s = _settings().injury
    rng = np.random.default_rng(7)
    for _ in range(100):
        context = _random_features(rng)
        no = assess_injury(InjuryFeatures.model_validate({**context, "mixed_action": False}), s)
        yes = assess_injury(InjuryFeatures.model_validate({**context, "mixed_action": True}), s)
        assert _component(yes, "lumbar") >= _component(no, "lumbar")
        assert yes.overall >= no.overall


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5.0, 0.0), (10.0, 0.0), (25.0, 50.0), (40.0, 100.0), (70.0, 100.0)],
)
def test_driver_score_counter_rotation_bounds(value: float, expected: float) -> None:
    low, high = _settings().injury.drivers["counter_rotation_deg"]  # 10 -> 40
    assert driver_score(value, low, high) == pytest.approx(expected)


def test_driver_score_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        driver_score(1.0, 5.0, 5.0)


def test_hand_computed_full_model() -> None:
    s = _settings().injury
    f = InjuryFeatures(
        counter_rotation_deg=25.0,  # 50
        mixed_action=True,  # 100
        lateral_flexion_deg=27.5,  # 50
        knee_collapse_deg=20.0,  # 50
        stride_pct=97.5,  # 50
        arm_slot_deg=40.0,  # 50
        arm_angular_velocity_deg_s=2250.0,  # 50
        trunk_rotation_speed_deg_s=700.0,  # 50
        fatigue_index=50.0,  # 50
    )
    risk = assess_injury(f, s)
    comps = {c.name: c.score for c in risk.components}
    assert comps["lumbar"] == pytest.approx((50 + 100 + 50) / 3)
    assert comps["knee"] == comps["shoulder"] == comps["side_strain"] == pytest.approx(50.0)
    assert comps["fatigue_modifier"] == pytest.approx(50.0)
    expected = 0.40 * 200 / 3 + 0.20 * 50 + 0.15 * 50 + 0.15 * 50 + 0.10 * 50
    assert risk.overall == pytest.approx(expected)
    assert risk.band == "medium"
    assert risk.notes == []
    assert risk.top_drivers[0].startswith("mixed action")
    assert len(risk.top_drivers) == 3


def test_missing_components_renormalise_and_note() -> None:
    s = _settings().injury
    risk = assess_injury(InjuryFeatures(knee_collapse_deg=40.0, stride_pct=110.0), s)
    assert [c.name for c in risk.components] == ["knee"]
    assert risk.overall == pytest.approx(100.0) and risk.band == "high"
    assert any("lumbar not assessed" in n for n in risk.notes)
    assert any("20%" in n for n in risk.notes)  # knee is 20% of the model


def test_no_inputs_raises() -> None:
    with pytest.raises(ValueError, match="No injury-risk input"):
        assess_injury(InjuryFeatures(), _settings().injury)


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.0, "low"),
        (34.9, "low"),
        (35.0, "medium"),
        (65.0, "medium"),
        (65.1, "high"),
        (100.0, "high"),
    ],
)
def test_bands(score: float, band: str) -> None:
    assert band_for(score, _settings().injury) == band


def test_top_drivers_unique_and_only_elevated() -> None:
    s = _settings().injury
    risk = assess_injury(InjuryFeatures(lateral_flexion_deg=40.0, knee_collapse_deg=0.0), s)
    # lateral flexion feeds lumbar and side_strain but is listed once; zero-risk drivers aren't
    assert len(risk.top_drivers) == 1 and risk.top_drivers[0].startswith("side bend")


def test_disclaimer_always_present() -> None:
    risk = assess_injury(InjuryFeatures(fatigue_index=10.0), _settings().injury)
    assert "not a trained medical model" in risk.disclaimer
    assert "not medical advice" in risk.disclaimer


def _raw_config() -> dict:  # type: ignore[type-arg]
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.pop("arm_slot_deg"), "drivers"),
        (lambda d: d.__setitem__("stride_pct", [110, 85]), "low < high"),
    ],
)
def test_driver_config_validation(tmp_path: Path, mutate: object, match: str) -> None:
    raw = _raw_config()
    mutate(raw["injury"]["drivers"])  # type: ignore[operator]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_settings(path)


# --- synthetic end-to-end ----------------------------------------------------------------


@cache
def _risk(**kw: object) -> tuple[InjuryRisk, InjuryFeatures]:
    s = _settings()
    r = generate(SynthParams.model_validate({"height_px": 540.0, **kw}))
    seq = clean_sequence(r.sequence, s.pose)
    raw = clean_sequence(r.sequence, s.pose, smooth=False)
    height = estimate_height_px(seq, s.body).height_px
    ph = detect_phases(seq, height, s.phases, raw=raw)
    al = assess_alignment(seq, ph, s.alignment, raw=raw)
    fr = assess_faults(seq, ph, height, s, raw=raw)
    feats = injury_features(seq, ph, al, fr, s, fatigue_index=0.0)
    return assess_injury(feats, s.injury), feats


@pytest.mark.parametrize("arm", ["right", "left"])
def test_synthetic_lumbar_rises_with_bad_alignment(arm: str) -> None:
    good, _ = _risk(
        view="front", bowling_arm=arm, lateral_flexion_deg=12.0, counter_rotation_deg=5.0
    )
    bad, feats = _risk(
        view="front",
        bowling_arm=arm,
        action_type="mixed",
        lateral_flexion_deg=38.0,
        counter_rotation_deg=40.0,
    )
    assert feats.mixed_action is True
    assert _component(bad, "lumbar") > _component(good, "lumbar") + 40
    assert bad.overall > good.overall and bad.band in ("medium", "high")


@pytest.mark.parametrize("arm", ["right", "left"])
def test_synthetic_knee_rises_with_collapse_and_overstride(arm: str) -> None:
    good, _ = _risk(bowling_arm=arm, knee_flex_change_deg=5.0, stride_pct=78.0)
    bad, _ = _risk(bowling_arm=arm, knee_flex_change_deg=35.0, stride_pct=100.0)
    assert _component(bad, "knee") > _component(good, "knee") + 50


def test_view_specific_features() -> None:
    _, side = _risk()
    _, front = _risk(view="front")
    assert side.arm_angular_velocity_deg_s is not None and side.arm_slot_deg is None
    assert side.counter_rotation_deg is None and side.mixed_action is None
    assert front.arm_slot_deg is not None and front.arm_angular_velocity_deg_s is None
    assert front.trunk_rotation_speed_deg_s is not None and front.knee_collapse_deg is None


def test_arm_slot_vertical_arm() -> None:
    s = _settings()
    r = generate(SynthParams(view="front"))
    seq = clean_sequence(r.sequence, s.pose)
    slot = arm_slot_deg(seq, r.phases)
    assert slot is not None and slot < 5.0  # the synthetic arm swings straight overhead
