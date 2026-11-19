"""Module E: fatigue drift within a session, and workload (acute:chronic ratio) across sessions.

Fatigue: the baseline is the mean of the first ``fatigue.baseline_deliveries`` deliveries. For
each later delivery, the % change of run-up speed, release height, peak wrist speed and
technique score is computed. Only *drops* count towards fatigue: a drop of
``full_drop_pct`` scores 100 for that metric, and a delivery's index is the weighted mean of
its metric scores. The session fatigue index is the mean over the last
``recent_deliveries``. An alert fires when that index exceeds ``alert_index``, or when a recent
delivery has at least ``fault_increase_alert`` more faults than the baseline average.

Workload: acute load = balls in the last ``acute_days``; chronic load = balls in the last
``chronic_days`` expressed per ``acute_days`` (a weekly average). ACWR = acute / chronic,
flagged above ``acwr_flag``. Without ``min_history_days`` of history the ratio isn't reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from bowlsense.config import FatigueSettings, Settings, WorkloadSettings
from bowlsense.geometry.angles import keypoint_track, linear_speed
from bowlsense.geometry.body import joint
from bowlsense.phases.heuristic import hip_speed
from bowlsense.schemas import (
    FatigueDrift,
    FatigueInput,
    FatigueResult,
    FaultReport,
    PhaseEvents,
    PoseSequence,
    WorkloadResult,
)

_INPUT_FIELD = {
    "runup_speed": "runup_speed_h_s",
    "release_height": "release_height_pct",
    "wrist_speed": "wrist_speed_h_s",
    "technique_score": "technique_score",
}


def delivery_fatigue_input(
    delivery_idx: int,
    seq: PoseSequence,
    phases: PhaseEvents,
    height_px: float,
    faults: FaultReport,
    settings: Settings,
) -> FatigueInput:
    """The numbers fatigue tracks for one delivery. Speeds are in body heights per second."""
    side = seq.view in ("side", "unknown")
    runup: float | None = None
    if side:
        speed = hip_speed(seq, settings.faults.runup_speed_smooth_s)[
            phases.run_up_start : phases.bfc + 1
        ]
        runup = float(np.nanmax(speed)) / height_px if np.isfinite(speed).any() else None
    wrist = linear_speed(keypoint_track(seq, joint(seq.bowling_arm, "bowling", "wrist")), seq.fps)
    hi = min(len(seq.frames) - 1, phases.release + round(0.1 * seq.fps))
    w = wrist[phases.bfc : hi + 1]
    return FatigueInput(
        delivery_idx=delivery_idx,
        runup_speed_h_s=runup,
        release_height_pct=faults.measurements.get("LOW_RELEASE"),
        wrist_speed_h_s=float(np.nanmax(w)) / height_px if np.isfinite(w).any() else None,
        technique_score=faults.technique_score,
        fault_count=len(faults.faults),
    )


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _pct_change(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return 100.0 * (value - base) / abs(base)


def drop_score(pct_change: float | None, full_drop_pct: float) -> float | None:
    """0-100 score for a metric's drop from baseline; rises never count as fatigue."""
    if pct_change is None:
        return None
    drop = max(0.0, -pct_change)
    return float(np.clip(drop / full_drop_pct, 0.0, 1.0) * 100.0)


def assess_fatigue(inputs: Sequence[FatigueInput], settings: FatigueSettings) -> FatigueResult:
    """Fatigue drift over a session's deliveries (see module docstring)."""
    ordered = sorted(inputs, key=lambda i: i.delivery_idx)
    n_base = settings.baseline_deliveries
    if len(ordered) <= n_base:
        return FatigueResult(
            baseline_deliveries=min(len(ordered), n_base),
            drifts=[],
            fatigue_index=0.0,
            alert=False,
            alert_reasons=[],
            notes=[
                f"Fatigue needs more than {n_base} deliveries (got {len(ordered)}); "
                "no fatigue trend can be measured yet."
            ],
        )
    base_rows, later = ordered[:n_base], ordered[n_base:]
    baseline = {
        metric: _mean([getattr(r, field) for r in base_rows])
        for metric, field in _INPUT_FIELD.items()
    }
    base_faults = float(np.mean([r.fault_count for r in base_rows]))

    drifts: list[FatigueDrift] = []
    indices: list[float | None] = []
    for row in later:
        changes = {
            metric: _pct_change(getattr(row, field), baseline[metric])
            for metric, field in _INPUT_FIELD.items()
        }
        scored = [
            (settings.weights[m], s)
            for m, c in changes.items()
            if (s := drop_score(c, settings.full_drop_pct[m])) is not None
        ]
        weight = sum(w for w, _ in scored)
        index = sum(w * s for w, s in scored) / weight if weight > 0 else None
        indices.append(index)
        drifts.append(
            FatigueDrift(
                delivery_idx=row.delivery_idx,
                runup_speed_pct=changes["runup_speed"],
                release_height_pct=changes["release_height"],
                wrist_speed_pct=changes["wrist_speed"],
                technique_score_pct=changes["technique_score"],
                fault_count_delta=round(row.fault_count - base_faults),
                fatigue_index=index,
            )
        )

    recent = indices[-settings.recent_deliveries :]
    recent_vals = [i for i in recent if i is not None]
    notes: list[str] = []
    if recent_vals:
        session_index = float(np.mean(recent_vals))
    else:
        session_index = 0.0
        notes.append("No fatigue metric could be measured on the most recent deliveries.")

    reasons: list[str] = []
    if session_index > settings.alert_index:
        reasons.append(
            f"fatigue index {session_index:.0f} is above {settings.alert_index:.0f}: speed, "
            "release height or technique has dropped from your first deliveries"
        )
    worst_rise = max(r.fault_count for r in later[-settings.recent_deliveries :]) - base_faults
    if worst_rise >= settings.fault_increase_alert:
        reasons.append(
            f"{worst_rise:.0f} more technique faults than in your first {n_base} deliveries"
        )
    return FatigueResult(
        baseline_deliveries=n_base,
        drifts=drifts,
        fatigue_index=float(np.clip(session_index, 0.0, 100.0)),
        alert=bool(reasons),
        alert_reasons=reasons,
        notes=notes,
    )


def assess_workload(
    history: Sequence[tuple[date, int]],
    today: date,
    balls_this_session: int,
    settings: WorkloadSettings,
) -> WorkloadResult:
    """Acute:chronic workload ratio for a bowler.

    ``history`` holds ``(day, balls)`` for earlier sessions; this session's balls are added
    on ``today``. Sessions after ``today`` are ignored.
    """
    if balls_this_session < 0:
        raise ValueError("balls_this_session must be non-negative")
    days = [(d, b) for d, b in history if d <= today] + [(today, balls_this_session)]
    first = min(d for d, _ in days)
    if (today - first).days < settings.min_history_days:
        return WorkloadResult(
            balls_this_session=balls_this_session,
            note=(
                f"Workload ratio needs at least {settings.min_history_days} days of history; "
                f"the first recorded session is {(today - first).days} days ago."
            ),
        )

    def load(window_days: int) -> int:
        start = today - timedelta(days=window_days - 1)
        return sum(b for d, b in days if start <= d <= today)

    acute = float(load(settings.acute_days))
    chronic = load(settings.chronic_days) * settings.acute_days / settings.chronic_days
    if chronic <= 0:
        return WorkloadResult(
            balls_this_session=balls_this_session,
            acute_load=acute,
            chronic_load=0.0,
            note="No balls bowled in the chronic window, so the ratio is undefined.",
        )
    acwr = acute / chronic
    flagged = acwr > settings.acwr_flag
    note = (
        f"Acute load is {acwr:.2f}x your recent weekly average (flag above "
        f"{settings.acwr_flag:.1f}). Sudden spikes in workload are linked to injury."
        if flagged
        else f"Acute load is {acwr:.2f}x your recent weekly average."
    )
    return WorkloadResult(
        balls_this_session=balls_this_session,
        acute_load=acute,
        chronic_load=chronic,
        acwr=acwr,
        flagged=flagged,
        note=note,
    )
