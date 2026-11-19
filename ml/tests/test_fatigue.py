"""Tests for analysis/fatigue.py (module E): fatigue drift gate, workload ACWR, ball history."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import cache
from itertools import pairwise

import numpy as np
import pytest
from sqlalchemy import insert

from bowlsense.analysis.fatigue import (
    assess_fatigue,
    assess_workload,
    delivery_fatigue_input,
    drop_score,
)
from bowlsense.analysis.faults import assess_faults
from bowlsense.config import Settings, load_settings
from bowlsense.db import models as m
from bowlsense.db.repo import Repository
from bowlsense.errors import NotFoundError
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import FatigueInput
from synth.skeleton_gen import SynthParams, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


def _row(i: int, **kw: float | int | None) -> FatigueInput:
    base: dict[str, float | int | None] = {
        "runup_speed_h_s": 3.2,
        "release_height_pct": 120.0,
        "wrist_speed_h_s": 12.0,
        "technique_score": 90.0,
        "fault_count": 1,
    }
    return FatigueInput.model_validate({"delivery_idx": i, **base, **kw})


def _session(later: Sequence[Mapping[str, float | int | None]]) -> list[FatigueInput]:
    return [_row(i) for i in range(6)] + [_row(6 + j, **kw) for j, kw in enumerate(later)]


# --- gate: fatigue drift ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "field", "full_drop", "base"),
    [
        ("runup_speed", "runup_speed_h_s", 15.0, 3.2),
        ("release_height", "release_height_pct", 5.0, 120.0),
        ("wrist_speed", "wrist_speed_h_s", 10.0, 12.0),
        ("technique_score", "technique_score", 20.0, 90.0),
    ],
)
def test_gate_each_metric_drift_is_measured(
    metric: str, field: str, full_drop: float, base: float
) -> None:
    """A drop of half the full-drop % in one metric gives that metric 50 and the delivery
    index weight x 50 (all other metrics unchanged)."""
    fs = _settings().fatigue
    drop = base * (1 - full_drop / 200.0)
    res = assess_fatigue(_session([{field: drop}]), fs)
    drift = res.drifts[0]
    assert getattr(drift, f"{metric}_pct") == pytest.approx(-full_drop / 2.0)
    total_w = sum(fs.weights.values())
    assert drift.fatigue_index == pytest.approx(fs.weights[metric] * 50.0 / total_w)


def test_gate_index_grows_with_progressive_drift() -> None:
    fs = _settings().fatigue
    later = [
        {
            "runup_speed_h_s": 3.2 * (1 - 0.02 * k),
            "release_height_pct": 120 - 0.8 * k,
            "wrist_speed_h_s": 12.0 * (1 - 0.015 * k),
            "technique_score": 90 - 2.5 * k,
        }
        for k in range(1, 9)
    ]
    res = assess_fatigue(_session(later), fs)
    indices = [d.fatigue_index for d in res.drifts if d.fatigue_index is not None]
    assert len(indices) == len(res.drifts)
    assert all(b > a for a, b in pairwise(indices))
    assert res.fatigue_index == pytest.approx(np.mean(indices[-3:]))
    assert res.alert and any("fatigue index" in r for r in res.alert_reasons)


def test_rises_never_count_as_fatigue() -> None:
    res = assess_fatigue(
        _session([{"runup_speed_h_s": 4.0, "release_height_pct": 130.0, "technique_score": 100.0}]),
        _settings().fatigue,
    )
    assert res.drifts[0].fatigue_index == pytest.approx(0.0)
    assert res.drifts[0].runup_speed_pct == pytest.approx(25.0)
    assert not res.alert


def test_missing_metrics_drop_out_of_the_weights() -> None:
    fs = _settings().fatigue
    res = assess_fatigue(
        _session([{"runup_speed_h_s": None, "wrist_speed_h_s": None, "release_height_pct": 114.0}]),
        fs,
    )
    d = res.drifts[0]
    assert d.runup_speed_pct is None and d.wrist_speed_pct is None
    w = fs.weights
    expected = w["release_height"] * 100.0 / (w["release_height"] + w["technique_score"])
    assert d.fatigue_index == pytest.approx(expected)


def test_fault_increase_alert() -> None:
    fs = _settings().fatigue
    res = assess_fatigue(_session([{"fault_count": 1}, {"fault_count": 3}]), fs)
    assert res.alert and any("more technique faults" in r for r in res.alert_reasons)
    assert res.drifts[1].fault_count_delta == 2
    calm = assess_fatigue(_session([{"fault_count": 2}]), fs)
    assert not calm.alert


def test_too_few_deliveries() -> None:
    res = assess_fatigue([_row(i) for i in range(6)], _settings().fatigue)
    assert res.drifts == [] and res.fatigue_index == 0.0 and not res.alert
    assert res.notes and "more than 6" in res.notes[0]


def test_input_order_doesnt_matter() -> None:
    rows = _session([{"technique_score": 70.0}, {"technique_score": 60.0}])
    fs = _settings().fatigue
    assert assess_fatigue(rows[::-1], fs) == assess_fatigue(rows, fs)


@pytest.mark.parametrize(
    ("change", "score"), [(None, None), (5.0, 0.0), (-5.0, 50.0), (-30.0, 100.0)]
)
def test_drop_score(change: float | None, score: float | None) -> None:
    assert drop_score(change, 10.0) == score


def test_synthetic_tiring_session() -> None:
    s = _settings()
    inputs = []
    for i in range(10):
        tired = max(0, i - 5)
        p = SynthParams(
            height_px=540,
            release_height_pct=120 - 1.5 * tired,
            knee_flex_change_deg=8 + 6 * tired,
        )
        r = generate(p)
        seq = clean_sequence(r.sequence, s.pose)
        raw = clean_sequence(r.sequence, s.pose, smooth=False)
        h = estimate_height_px(seq, s.body).height_px
        ph = detect_phases(seq, h, s.phases, raw=raw)
        inputs.append(
            delivery_fatigue_input(i, seq, ph, h, assess_faults(seq, ph, h, s, raw=raw), s)
        )
    res = assess_fatigue(inputs, s.fatigue)
    indices = [d.fatigue_index for d in res.drifts]
    assert indices[-1] is not None and indices[0] is not None and indices[-1] > indices[0] + 10
    assert res.drifts[-1].release_height_pct is not None
    assert res.drifts[-1].release_height_pct < -2.0
    assert res.alert


# --- workload -----------------------------------------------------------------------------------

TODAY = date(2026, 11, 20)


def test_workload_hand_computed() -> None:
    ws = _settings().workload
    history = [(TODAY - timedelta(days=d), 30) for d in (27, 20, 13, 6, 3)]
    res = assess_workload(history, TODAY, 40, ws)
    # acute (last 7 days incl. today): 30 + 30 + 40 = 100; chronic: 190 balls / 4 weeks
    assert res.acute_load == pytest.approx(100.0)
    assert res.chronic_load == pytest.approx(190 * 7 / 28)
    assert res.acwr == pytest.approx(100 / 47.5)
    assert res.flagged and "linked to injury" in res.note


def test_workload_steady_not_flagged() -> None:
    history = [(TODAY - timedelta(days=d), 20) for d in range(28, 0, -2)]
    res = assess_workload(history, TODAY, 20, _settings().workload)
    assert res.acwr is not None and 0.8 < res.acwr < 1.3 and not res.flagged


def test_workload_needs_history() -> None:
    res = assess_workload([(TODAY - timedelta(days=10), 30)], TODAY, 30, _settings().workload)
    assert res.acwr is None and res.acute_load is None and not res.flagged
    assert "21 days" in res.note and res.balls_this_session == 30


def test_workload_ignores_future_and_old_sessions() -> None:
    ws = _settings().workload
    history = [(TODAY - timedelta(days=40), 50), (TODAY + timedelta(days=2), 999)]
    res = assess_workload(history, TODAY, 0, ws)
    assert res.acwr is None and res.chronic_load == 0.0 and "undefined" in res.note


def test_workload_rejects_negative_balls() -> None:
    with pytest.raises(ValueError):
        assess_workload([], TODAY, -1, _settings().workload)


# --- repository: ball history -------------------------------------------------------------------


def _session_with_deliveries(
    repo: Repository, bowler_id: str, balls: int, when: datetime | None, status: str = "done"
) -> str:
    s = repo.create_session("raw/x.mp4", bowler_id=bowler_id, recorded_at=when)
    repo.set_session_status(s.id, status)  # type: ignore[arg-type]
    with repo.engine.begin() as conn:
        for i in range(balls):
            conn.execute(
                insert(m.Delivery).values(
                    id=m.new_id(), created_at=m.utcnow(), session_id=s.id, idx=i
                )
            )
    return s.id


def test_bowler_ball_history(repo: Repository) -> None:
    b = repo.create_bowler("A")
    other = repo.create_bowler("B")
    d1 = datetime(2026, 11, 1, 10, tzinfo=UTC)
    d2 = datetime(2026, 11, 10, 10, tzinfo=UTC)
    _session_with_deliveries(repo, b.id, 6, d2)
    _session_with_deliveries(repo, b.id, 4, d1)
    current = _session_with_deliveries(repo, b.id, 3, None)
    _session_with_deliveries(repo, b.id, 9, d1, status="failed")  # not counted
    _session_with_deliveries(repo, other.id, 7, d1)  # other bowler
    history = repo.bowler_ball_history(b.id, exclude_session_id=current)
    assert history == [(date(2026, 11, 1), 4), (date(2026, 11, 10), 6)]
    with_current = repo.bowler_ball_history(b.id)
    assert len(with_current) == 3 and sum(n for _, n in with_current) == 13
    with pytest.raises(NotFoundError):
        repo.bowler_ball_history("missing")
