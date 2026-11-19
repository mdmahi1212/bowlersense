"""Module C: technique faults and the technique score.

Each fault measures one value from the camera view it needs, compares it with its threshold
(``config/default.yaml -> faults``) and, when past it, reports a ``Fault`` with a severity
that ramps linearly from 0 at the threshold to 100 at the full-severity point (2 x threshold
unless ``faults.severity_full`` sets one).

``technique_score = 100 - weighted mean severity`` over the faults that could be measured
(unfired ones count as 0). Faults that need the other camera view are listed in
``FaultReport.not_measured``, never scored as zero.

Measurement definitions shared with the synthetic generator (see PROGRESS.md):
- stride = back ankle at BFC to front ankle at FFC, % of height;
- release height = estimated ball height = wrist height above the ground + hand length.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bowlsense.config import FaultSettings, Settings
from bowlsense.errors import InvalidInputError, PoseQualityError
from bowlsense.geometry.angles import (
    FloatArray,
    joint_angle,
    keypoint_track,
    linear_speed,
    midpoint,
)
from bowlsense.geometry.body import joint
from bowlsense.phases.heuristic import hip_speed
from bowlsense.pose.smoothing import savgol_derivative
from bowlsense.schemas import Fault, FaultReport, PhaseEvents, PoseSequence

ViewGroup = Literal["side", "front"]
SIDE_VIEWS = frozenset({"side", "unknown"})


@dataclass(frozen=True)
class FaultContext:
    """Everything a fault measurement needs."""

    seq: PoseSequence
    phases: PhaseEvents
    height_px: float
    settings: Settings
    raw: PoseSequence | None = None

    def track(self, role: str, part: str) -> FloatArray:
        """Keypoint track for a role-relative joint (e.g. front ankle)."""
        return keypoint_track(self.seq, joint(self.seq.bowling_arm, role, part))  # type: ignore[arg-type]

    def pct(self, distance_px: float) -> float:
        """Pixels to % of standing height."""
        return 100.0 * distance_px / self.height_px


Measurement = tuple[float, list[int]]  # value, frames involved


def _finite(value: float, what: str) -> float:
    if not np.isfinite(value):
        raise PoseQualityError(f"{what} not visible")
    return float(value)


def _knee_collapse(c: FaultContext) -> Measurement:
    """Front-knee flexion added between FFC and release.

    The knee changes sharply at foot strike and keeps bending after release, so smoothed
    keypoints would leak both into the window. Angles come from the unsmoothed track (when
    given) and a low-order polynomial is fitted strictly inside [FFC, release].
    """
    p = c.phases
    src = c.raw if c.raw is not None else c.seq
    arm = c.seq.bowling_arm
    angle = joint_angle(
        *(keypoint_track(src, joint(arm, "front", part)) for part in ("hip", "knee", "ankle"))
    )
    t = np.arange(p.ffc, p.release + 1, dtype=np.float64)
    seg = angle[p.ffc : p.release + 1]
    ok = np.isfinite(seg)
    degree = c.settings.faults.knee_fit_degree
    if int(ok.sum()) <= degree:
        raise PoseQualityError("front knee not visible between FFC and release")
    coeffs = np.polyfit(t[ok] - p.ffc, seg[ok], degree)
    at_ffc = float(np.polyval(coeffs, 0.0))
    at_rel = float(np.polyval(coeffs, float(p.release - p.ffc)))
    return at_ffc - at_rel, [p.ffc, p.release]


def stride_pct(c: FaultContext) -> Measurement:
    """Back ankle at BFC to front ankle at FFC, horizontal, % of height."""
    back = _finite(c.track("back", "ankle")[c.phases.bfc, 0], "back ankle at BFC")
    front = _finite(c.track("front", "ankle")[c.phases.ffc, 0], "front ankle at FFC")
    return c.pct(abs(front - back)), [c.phases.bfc, c.phases.ffc]


def _head_fall(c: FaultContext) -> Measurement:
    r = c.phases.release
    nose = _finite(keypoint_track(c.seq, "nose")[r, 0], "nose at release")
    foot = _finite(c.track("front", "ankle")[r, 0], "front ankle at release")
    return c.pct(abs(nose - foot)), [r]


def _lateral_flexion(c: FaultContext) -> Measurement:
    r = c.phases.release
    sh = midpoint(keypoint_track(c.seq, "l_shoulder"), keypoint_track(c.seq, "r_shoulder"))[r]
    hp = midpoint(keypoint_track(c.seq, "l_hip"), keypoint_track(c.seq, "r_hip"))[r]
    dx, dy_up = float(sh[0] - hp[0]), float(hp[1] - sh[1])
    if not (np.isfinite(dx) and np.isfinite(dy_up)):
        raise PoseQualityError("trunk not visible at release")
    return float(np.degrees(np.arctan2(abs(dx), dy_up))), [r]


def _weak_front_arm(c: FaultContext) -> Measurement:
    f = c.phases.ffc
    wrist = _finite(c.track("non_bowling", "wrist")[f, 1], "front wrist at FFC")
    shoulder = _finite(c.track("non_bowling", "shoulder")[f, 1], "front shoulder at FFC")
    return c.pct(wrist - shoulder), [f]  # image y grows downward: positive = below


def _runup_decel(c: FaultContext) -> Measurement:
    p, fps = c.phases, c.seq.fps
    speed = hip_speed(c.seq, c.settings.faults.runup_speed_smooth_s)
    run = speed[p.run_up_start : p.bfc + 1]
    if not np.isfinite(run).any():
        raise PoseQualityError("hips not visible in the run-up")
    peak = float(np.nanmax(run))
    if peak <= 0:
        raise PoseQualityError("no run-up movement")
    lo = max(p.run_up_start, p.bfc - round(c.settings.faults.runup_final_window_s * fps))
    window = speed[lo : p.bfc + 1]
    return 100.0 * (peak - float(np.nanmin(window))) / peak, [lo, p.bfc]


def release_height_pct(c: FaultContext) -> Measurement:
    """Estimated ball height at release (wrist above ground + hand length), % of height."""
    p, body = c.phases, c.settings.body
    ankle = c.track("front", "ankle")[p.ffc : p.release + 1, 1]
    if not np.isfinite(ankle).any():
        raise PoseQualityError("front ankle not visible between FFC and release")
    ground = float(np.nanmedian(ankle)) + body.ankle_height_pct / 100.0 * c.height_px
    wrist = _finite(c.track("bowling", "wrist")[p.release, 1], "bowling wrist at release")
    return c.pct(ground - wrist) + body.hand_length_pct, [p.release]


def _chain_order(c: FaultContext) -> Measurement:
    p, fps = c.phases, c.seq.fps
    hi = min(
        len(c.seq.frames) - 1,
        p.release + round(c.settings.faults.chain_window_after_release_s * fps),
    )
    window_s = c.settings.pose.savgol_window_s

    def peak(values: FloatArray, what: str) -> int:
        seg = np.abs(values[p.bfc : hi + 1])
        if not np.isfinite(seg).any():
            raise PoseQualityError(f"{what} not visible from BFC to release")
        return p.bfc + int(np.nanargmax(seg))

    def width_rate(part: str) -> FloatArray:
        width = keypoint_track(c.seq, f"l_{part}")[:, 0] - keypoint_track(c.seq, f"r_{part}")[:, 0]
        return savgol_derivative(width, fps, window_s)

    t_pelvis = peak(width_rate("hip"), "hips")
    t_trunk = peak(width_rate("shoulder"), "shoulders")
    t_arm = peak(linear_speed(c.track("bowling", "wrist"), fps), "bowling wrist")
    lag_frames = max(0, t_pelvis - t_trunk, t_trunk - t_arm, t_pelvis - t_arm)
    return lag_frames / fps, [t_pelvis, t_trunk, t_arm]


def _short_follow(c: FaultContext) -> Measurement:
    p = c.phases
    hips = midpoint(keypoint_track(c.seq, "l_hip"), keypoint_track(c.seq, "r_hip"))[:, 0]
    travel = abs(
        _finite(hips[p.follow_through_end], "hips at follow-through end")
        - _finite(hips[p.release], "hips at release")
    )
    stride_value, _ = stride_pct(c)
    stride_px = stride_value / 100.0 * c.height_px
    if stride_px <= 0:
        raise PoseQualityError("stride length is zero")
    return 100.0 * travel / stride_px, [p.release, p.follow_through_end]


@dataclass(frozen=True)
class FaultSpec:
    """How one fault is measured and judged."""

    code: str
    name: str
    view: ViewGroup
    above: bool  # True: fires when value > threshold; False: when value < threshold
    threshold: Callable[[FaultSettings], float]
    measure: Callable[[FaultContext], Measurement]
    explain: Callable[[float, float], str]


FAULT_SPECS: tuple[FaultSpec, ...] = (
    FaultSpec(
        "KNEE_COLLAPSE",
        "Front knee collapse",
        "side",
        True,
        lambda s: s.knee_collapse_deg,
        _knee_collapse,
        lambda v, t: (
            f"Your front knee bent a further {v:.0f} deg between front-foot contact and "
            f"release (limit {t:.0f} deg). A firmer front leg passes more energy into the ball."
        ),
    ),
    FaultSpec(
        "OVERSTRIDE",
        "Over-long delivery stride",
        "side",
        True,
        lambda s: s.overstride_pct,
        stride_pct,
        lambda v, t: (
            f"Your delivery stride was {v:.0f}% of your height (limit {t:.0f}%). A very long "
            "stride makes it hard to brace the front leg and loads the knee and back."
        ),
    ),
    FaultSpec(
        "UNDERSTRIDE",
        "Short delivery stride",
        "side",
        False,
        lambda s: s.understride_pct,
        stride_pct,
        lambda v, t: (
            f"Your delivery stride was {v:.0f}% of your height (target at least {t:.0f}%). "
            "A short stride limits how much run-up speed carries into the ball."
        ),
    ),
    FaultSpec(
        "HEAD_FALL",
        "Head falling away",
        "front",
        True,
        lambda s: s.head_fall_pct,
        _head_fall,
        lambda v, t: (
            f"At release your head was {v:.0f}% of your height to the side of your front foot "
            f"(limit {t:.0f}%). Keeping the head over the front foot helps accuracy."
        ),
    ),
    FaultSpec(
        "LATERAL_FLEXION",
        "Excessive side bend",
        "front",
        True,
        lambda s: s.lateral_flexion_deg,
        _lateral_flexion,
        lambda v, t: (
            f"Your trunk was bent {v:.0f} deg sideways at release (limit {t:.0f} deg). Large "
            "side bend is linked to lower-back stress in fast bowlers."
        ),
    ),
    FaultSpec(
        "WEAK_FRONT_ARM",
        "Weak front arm",
        "side",
        True,
        lambda s: s.weak_front_arm_pct,
        _weak_front_arm,
        lambda v, t: (
            f"At front-foot contact your front wrist was {v:.0f}% of your height below your "
            "shoulder. A high, strong front arm helps you rotate and stay aligned."
        ),
    ),
    FaultSpec(
        "RUNUP_DECEL",
        "Slowing down in the run-up",
        "side",
        True,
        lambda s: s.runup_decel_pct,
        _runup_decel,
        lambda v, t: (
            f"Your speed dropped {v:.0f}% from its peak over your last strides (limit "
            f"{t:.0f}%). Keep building speed smoothly into the crease."
        ),
    ),
    FaultSpec(
        "LOW_RELEASE",
        "Low release point",
        "side",
        False,
        lambda s: s.low_release_pct,
        release_height_pct,
        lambda v, t: (
            f"The ball left your hand at about {v:.0f}% of your height (target at least "
            f"{t:.0f}%). A taller, more upright action raises the release and adds bounce."
        ),
    ),
    FaultSpec(
        "CHAIN_ORDER",
        "Out-of-sequence rotation",
        "side",
        True,
        lambda s: s.chain_order_tolerance_s,
        _chain_order,
        lambda v, t: (
            f"Your hips, trunk and arm didn't reach their fastest rotation in order (out by "
            f"{v * 1000:.0f} ms). Power should flow hips first, then trunk, then arm."
        ),
    ),
    FaultSpec(
        "SHORT_FOLLOW",
        "Cut-off follow-through",
        "side",
        False,
        lambda s: s.short_follow_pct,
        _short_follow,
        lambda v, t: (
            f"Your hips travelled only {v:.0f}% of your stride length after release (target "
            f"at least {t:.0f}%). Let the body carry through to slow down gradually."
        ),
    ),
)


def severity(spec: FaultSpec, value: float, settings: FaultSettings) -> float | None:
    """Severity 0-100 if the fault fires, else ``None``."""
    t = spec.threshold(settings)
    full = settings.severity_full.get(spec.code, 2.0 * t)
    if spec.above:
        if value <= t:
            return None
        span = full - t
    else:
        if value >= t:
            return None
        span = t - full
    if span <= 0:
        raise InvalidInputError(f"Severity ramp for {spec.code} is empty; check config.")
    excess = (value - t) if spec.above else (t - value)
    return float(np.clip(excess / span, 0.0, 1.0) * 100.0)


def _view_group(view: str) -> ViewGroup:
    return "side" if view in SIDE_VIEWS else "front"


def assess_faults(
    seq: PoseSequence,
    phases: PhaseEvents,
    height_px: float,
    settings: Settings,
    raw: PoseSequence | None = None,
) -> FaultReport:
    """Measure every fault this camera view allows and score the technique.

    ``raw`` is the same clip gated and gap-filled but not smoothed; fast-changing angles
    (front knee) are read from it when given.

    Raises:
        InvalidInputError: if ``height_px`` is not positive.
    """
    if not np.isfinite(height_px) or height_px <= 0:
        raise InvalidInputError(f"height_px must be positive, got {height_px}.")
    ctx = FaultContext(seq=seq, phases=phases, height_px=height_px, settings=settings, raw=raw)
    group = _view_group(seq.view)
    faults: list[Fault] = []
    measurements: dict[str, float] = {}
    not_measured: dict[str, str] = {}
    weighted, total_weight = 0.0, 0.0
    for spec in FAULT_SPECS:
        if spec.view != group:
            not_measured[spec.code] = f"needs {spec.view} view"
            continue
        try:
            value, frames = spec.measure(ctx)
        except PoseQualityError as exc:
            not_measured[spec.code] = str(exc)
            continue
        measurements[spec.code] = value
        weight = settings.faults.weights[spec.code]
        total_weight += weight
        sev = severity(spec, value, settings.faults)
        if sev is None:
            continue
        weighted += weight * sev
        threshold = spec.threshold(settings.faults)
        faults.append(
            Fault(
                code=spec.code,
                name=spec.name,
                severity=sev,
                value=value,
                threshold=threshold,
                frames=frames,
                explanation=spec.explain(value, threshold),
            )
        )
    if total_weight == 0:
        raise PoseQualityError(
            "No technique fault could be measured from this clip; check that the bowler's "
            "whole body is visible through the delivery stride."
        )
    score = 100.0 - weighted / total_weight
    return FaultReport(
        faults=faults,
        measurements=measurements,
        not_measured=not_measured,
        technique_score=float(np.clip(score, 0.0, 100.0)),
    )
