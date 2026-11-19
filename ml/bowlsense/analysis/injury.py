"""Module D: transparent, rule-based injury-risk screening.

NOT a trained medical model and NOT medical advice: ``InjuryRisk.disclaimer`` says so, and
the UI and PDF must show it.

Each driver scores 0 at or below its ``low`` bound and 100 at or above its ``high`` bound
(linear in between; ``config -> injury.drivers``). A component's score is the mean of the
drivers it could measure. The overall score is the weighted sum of components (weights from
``injury.weights``), renormalised over the components that had any data; the ones that
didn't are listed in ``notes`` rather than counted as zero risk.

| component        | drivers                                                    |
|------------------|------------------------------------------------------------|
| lumbar           | counter-rotation, mixed action (yes = 100), lateral flexion |
| knee             | knee collapse, stride length                               |
| shoulder         | arm slot (angle from vertical), peak arm angular velocity   |
| side_strain      | lateral flexion, trunk rotation speed                      |
| fatigue_modifier | session fatigue index (module E)                           |
"""

from __future__ import annotations

import numpy as np

from bowlsense.analysis.alignment import FRONTAL_VIEWS, full_width, yaw_series
from bowlsense.config import InjurySettings, Settings
from bowlsense.geometry.angles import keypoint_track, segment_angle
from bowlsense.geometry.body import joint
from bowlsense.pose.smoothing import savgol_derivative
from bowlsense.schemas import (
    AlignmentResult,
    FaultReport,
    InjuryFeatures,
    InjuryRisk,
    PhaseEvents,
    PoseSequence,
    RiskBand,
    RiskComponent,
)

COMPONENT_DRIVERS: dict[str, tuple[str, ...]] = {
    "lumbar": ("counter_rotation_deg", "mixed_action", "lateral_flexion_deg"),
    "knee": ("knee_collapse_deg", "stride_pct"),
    "shoulder": ("arm_slot_deg", "arm_angular_velocity_deg_s"),
    "side_strain": ("lateral_flexion_deg", "trunk_rotation_speed_deg_s"),
    "fatigue_modifier": ("fatigue_index",),
}

_DESCRIBE: dict[str, tuple[str, str]] = {
    # driver -> (plain words, unit format)
    "counter_rotation_deg": ("shoulder counter-rotation", "{:.0f} deg"),
    "mixed_action": ("mixed action (hips and shoulders misaligned)", ""),
    "lateral_flexion_deg": ("side bend at release", "{:.0f} deg"),
    "knee_collapse_deg": ("front-knee collapse", "{:.0f} deg"),
    "stride_pct": ("stride length", "{:.0f}% of height"),
    "arm_slot_deg": ("round-arm slot (arm angle from vertical)", "{:.0f} deg"),
    "arm_angular_velocity_deg_s": ("arm speed", "{:.0f} deg/s"),
    "trunk_rotation_speed_deg_s": ("trunk rotation speed", "{:.0f} deg/s"),
    "fatigue_index": ("fatigue", "index {:.0f}"),
}


def driver_score(value: float, low: float, high: float) -> float:
    """0 at or below ``low``, 100 at or above ``high``, linear in between (never decreasing)."""
    if not low < high:
        raise ValueError("driver bounds need low < high")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100.0)


def band_for(score: float, settings: InjurySettings) -> RiskBand:
    """Low below ``bands.low``, high above ``bands.high``, medium otherwise."""
    if score < settings.bands.low:
        return "low"
    if score > settings.bands.high:
        return "high"
    return "medium"


def _score_driver(name: str, features: InjuryFeatures, settings: InjurySettings) -> float | None:
    value = getattr(features, name)
    if value is None:
        return None
    if name == "mixed_action":
        return 100.0 if value else 0.0
    low, high = settings.drivers[name]
    return driver_score(float(value), low, high)


def _describe(name: str, features: InjuryFeatures, score: float) -> str:
    words, unit = _DESCRIBE[name]
    value = getattr(features, name)
    shown = "" if name == "mixed_action" else f" {unit.format(value)}"
    return f"{words}{shown} (risk {score:.0f}/100)"


def assess_injury(features: InjuryFeatures, settings: InjurySettings) -> InjuryRisk:
    """Score injury risk from measured features (see module docstring)."""
    components: list[RiskComponent] = []
    ranked: list[tuple[float, str]] = []
    notes: list[str] = []
    weighted, weight_total = 0.0, 0.0
    for name, drivers in COMPONENT_DRIVERS.items():
        scores: list[float] = []
        described: list[str] = []
        for driver in drivers:
            score = _score_driver(driver, features, settings)
            if score is None:
                continue
            scores.append(score)
            text = _describe(driver, features, score)
            described.append(text)
            ranked.append((score, text))
        if not scores:
            notes.append(f"{name} not assessed: none of its inputs could be measured")
            continue
        comp_score = float(np.mean(scores))
        weight = settings.weights[name]
        components.append(
            RiskComponent(name=name, score=comp_score, weight=weight, drivers=described)
        )
        weighted += weight * comp_score
        weight_total += weight
    if weight_total == 0:
        raise ValueError("No injury-risk input could be measured; cannot score risk.")
    overall = weighted / weight_total
    if weight_total < 1.0 - 1e-9:
        notes.append(
            f"overall score uses {weight_total:.0%} of the model; the rest needs inputs this "
            "clip doesn't provide (often the other camera view)"
        )
    unique: dict[str, float] = {}
    for score, text in ranked:  # lateral flexion feeds two components: list it once
        unique[text] = max(score, unique.get(text, 0.0))
    top = [t for t, s in sorted(unique.items(), key=lambda kv: -kv[1]) if s > 0][:3]
    return InjuryRisk(
        overall=float(overall),
        band=band_for(overall, settings),
        components=components,
        top_drivers=top,
        notes=notes,
    )


# --- measuring the extra drivers -------------------------------------------------------------


def arm_slot_deg(seq: PoseSequence, phases: PhaseEvents) -> float | None:
    """Bowling arm (shoulder to wrist) angle from vertical at release, from a front/behind
    camera (0 = straight up, larger = more round-arm). ``None`` from other views."""
    if seq.view not in FRONTAL_VIEWS:
        return None
    arm = seq.bowling_arm
    sh = keypoint_track(seq, joint(arm, "bowling", "shoulder"))[phases.release]
    wr = keypoint_track(seq, joint(arm, "bowling", "wrist"))[phases.release]
    if not (np.isfinite(sh).all() and np.isfinite(wr).all()):
        return None
    dx, dy_up = float(wr[0] - sh[0]), float(sh[1] - wr[1])
    return float(np.degrees(np.arctan2(abs(dx), dy_up)))


def arm_angular_velocity_deg_s(
    seq: PoseSequence, phases: PhaseEvents, window_s: float
) -> float | None:
    """Peak upper-arm angular velocity (deg/s) from BFC to just after release, side view."""
    if seq.view not in ("side", "unknown"):
        return None
    arm = seq.bowling_arm
    angle = segment_angle(
        keypoint_track(seq, joint(arm, "bowling", "shoulder")),
        keypoint_track(seq, joint(arm, "bowling", "elbow")),
    )
    unwrapped = np.full(angle.shape, np.nan)
    ok = np.isfinite(angle)
    if ok.sum() < 3:
        return None
    unwrapped[ok] = np.degrees(np.unwrap(np.radians(angle[ok])))
    rate = savgol_derivative(unwrapped, seq.fps, window_s)
    hi = min(len(seq.frames) - 1, phases.release + round(window_s * seq.fps))
    seg = np.abs(rate[phases.bfc : hi + 1])
    return float(np.nanmax(seg)) if np.isfinite(seg).any() else None


def trunk_rotation_speed_deg_s(
    seq: PoseSequence, phases: PhaseEvents, settings: Settings
) -> float | None:
    """Peak shoulder-line rotation speed (deg/s) from BFC to release, front/behind view."""
    if seq.view not in FRONTAL_VIEWS:
        return None
    full = full_width(seq, phases, "shoulder", settings.alignment.width_calib_top_frac)
    rate = savgol_derivative(
        yaw_series(seq, "shoulder", full), seq.fps, settings.pose.savgol_window_s
    )
    seg = np.abs(rate[phases.bfc : phases.release + 1])
    return float(np.nanmax(seg)) if np.isfinite(seg).any() else None


def injury_features(
    seq: PoseSequence,
    phases: PhaseEvents,
    alignment: AlignmentResult,
    faults: FaultReport,
    settings: Settings,
    fatigue_index: float | None = None,
) -> InjuryFeatures:
    """Collect every driver this clip can provide from the module A-C results."""
    m = faults.measurements
    return InjuryFeatures(
        counter_rotation_deg=alignment.counter_rotation_deg,
        mixed_action=(
            None if alignment.action_type == "unknown" else alignment.action_type == "mixed"
        ),
        lateral_flexion_deg=m.get("LATERAL_FLEXION"),
        knee_collapse_deg=m.get("KNEE_COLLAPSE"),
        stride_pct=m.get("OVERSTRIDE"),
        arm_slot_deg=arm_slot_deg(seq, phases),
        arm_angular_velocity_deg_s=arm_angular_velocity_deg_s(
            seq, phases, settings.pose.savgol_window_s
        ),
        trunk_rotation_speed_deg_s=trunk_rotation_speed_deg_s(seq, phases, settings),
        fatigue_index=fatigue_index,
    )
