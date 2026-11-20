"""What the coach sees: a compact metrics summary (never video or images) and the prompts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from bowlsense.config import Settings
from bowlsense.schemas import ConsistencyResult, DeliveryAnalysis, FatigueResult

UNITS: dict[str, str] = {
    "KNEE_COLLAPSE": "deg",
    "OVERSTRIDE": "% of height",
    "UNDERSTRIDE": "% of height",
    "HEAD_FALL": "% of height",
    "LATERAL_FLEXION": "deg",
    "WEAK_FRONT_ARM": "% of height below the shoulder",
    "RUNUP_DECEL": "% speed drop",
    "LOW_RELEASE": "% of height",
    "CHAIN_ORDER": "s out of order",
    "SHORT_FOLLOW": "% of stride",
}


class IssueSummary(BaseModel):
    """One technique fault aggregated over the session."""

    code: str
    name: str
    deliveries_affected: int
    mean_value: float
    threshold: float
    unit: str
    mean_severity: float
    explanation: str


class CoachSummary(BaseModel):
    """Everything the coach (LLM or template) is told about the session."""

    deliveries: int
    view: str
    bowling_arm: str
    technique_score_mean: float
    top_issues: list[IssueSummary]
    legality_flagged: int
    legality_screened: int  # deliveries the clip allowed a trustworthy elbow measurement for
    legality_max_extension_deg: float
    legality_threshold_deg: float
    legality_note: str
    injury_overall_mean: float
    injury_band_worst: str
    injury_top_drivers: list[str]
    injury_disclaimer: str
    ball_speeds_kmh: list[float]
    lengths: list[str]
    lines: list[str]
    delivery_types: list[str]
    fatigue_index: float
    fatigue_alerts: list[str]
    consistency_score: float | None
    closest_reference: str | None


_BAND_ORDER = {"low": 0, "medium": 1, "high": 2}


def summarise_session(
    deliveries: Sequence[DeliveryAnalysis],
    fatigue: FatigueResult,
    consistency: ConsistencyResult,
    view: str,
    bowling_arm: str,
    settings: Settings,
) -> CoachSummary:
    """Aggregate a session into a ``CoachSummary``.

    Issues are ranked by total weighted severity across deliveries; the top
    ``coach.top_issues`` are kept.

    Raises:
        ValueError: with no deliveries.
    """
    if not deliveries:
        raise ValueError("No deliveries to summarise.")
    by_code: dict[str, list[tuple[float, float, str, float, str]]] = defaultdict(list)
    for d in deliveries:
        for f in d.faults.faults:
            by_code[f.code].append((f.value, f.severity, f.name, f.threshold, f.explanation))
    weights = settings.faults.weights
    ranked = sorted(
        by_code.items(), key=lambda kv: -weights.get(kv[0], 1.0) * sum(x[1] for x in kv[1])
    )
    issues = []
    for code, rows in ranked[: settings.coach.top_issues]:
        worst = max(rows, key=lambda r: r[1])
        issues.append(
            IssueSummary(
                code=code,
                name=worst[2],
                deliveries_affected=len(rows),
                mean_value=float(np.mean([r[0] for r in rows])),
                threshold=worst[3],
                unit=UNITS.get(code, ""),
                mean_severity=float(np.mean([r[1] for r in rows])),
                explanation=worst[4],
            )
        )
    worst_injury = max(deliveries, key=lambda d: d.injury.overall)
    screened = [d for d in deliveries if d.legality.screened]
    sims: dict[str, list[float]] = defaultdict(list)
    for d in deliveries:
        for c in d.comparisons:
            sims[c.name].append(c.similarity_pct)
    closest = max(sims, key=lambda k: float(np.mean(sims[k]))) if sims else None
    return CoachSummary(
        deliveries=len(deliveries),
        view=view,
        bowling_arm=bowling_arm,
        technique_score_mean=float(np.mean([d.faults.technique_score for d in deliveries])),
        top_issues=issues,
        legality_flagged=sum(d.legality.flagged for d in deliveries),
        legality_screened=len(screened),
        legality_max_extension_deg=max(
            (d.legality.elbow_extension_deg for d in deliveries), default=0.0
        ),
        legality_threshold_deg=deliveries[0].legality.threshold_deg,
        legality_note=(
            screened[0].legality.note
            if screened
            else "Unverified: this clip can't support a trustworthy elbow measurement "
            "(needs a side-on camera at 60+ fps with the delivery phases found), so the "
            "verdict is a rough guide from the measured angle."
        ),
        injury_overall_mean=float(np.mean([d.injury.overall for d in deliveries])),
        injury_band_worst=max((d.injury.band for d in deliveries), key=_BAND_ORDER.__getitem__),
        injury_top_drivers=worst_injury.injury.top_drivers,
        injury_disclaimer=worst_injury.injury.disclaimer,
        ball_speeds_kmh=[d.ball.speed_kmh for d in deliveries if d.ball and d.ball.speed_kmh],
        lengths=[d.ball.length for d in deliveries if d.ball and d.ball.length],
        lines=[d.ball.line for d in deliveries if d.ball and d.ball.line],
        delivery_types=[d.delivery_type.delivery_type for d in deliveries if d.delivery_type],
        fatigue_index=fatigue.fatigue_index,
        fatigue_alerts=fatigue.alert_reasons,
        consistency_score=consistency.score,
        closest_reference=closest,
    )


def system_prompt(max_words: int, top: int) -> str:
    """The coach persona (spec Section 7H)."""
    return (
        "You are a friendly, encouraging cricket bowling coach. You are given ONLY measured "
        "metrics from a video analysis (JSON). Write a concrete coaching report: prioritise "
        f"the top {top} issues, cite the actual numbers, and give practical drills or cues for "
        "each. Do not make any medical diagnosis or claim an injury; describe injury-risk "
        "figures only as screening estimates and suggest seeing a professional about pain. "
        "Describe elbow legality as 2D screening only (the official test uses 3D motion "
        "capture). If legality_screened is 0, say the verdict is unverified for this "
        "clip and why; never quote an elbow angle or call the action illegal. "
        f"Use plain language a player understands. At most {max_words} words."
    )


def report_request(summary: CoachSummary) -> str:
    """The user message asking for the report."""
    return (
        "Session metrics (JSON):\n"
        f"{summary.model_dump_json(indent=1)}\n\n"
        "Write the coaching report now."
    )


def chat_context(summary: CoachSummary) -> str:
    """System context for the chat: the persona plus the session's metrics."""
    persona = system_prompt(150, 3).replace(
        "Write a concrete coaching report", "Answer the player's questions about this session"
    )
    return persona + "\n\nSession metrics (JSON):\n" + summary.model_dump_json(indent=1)
