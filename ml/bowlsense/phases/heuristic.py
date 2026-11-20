"""Rule-based delivery phase detection (spec Section 6). Always available.

Input is a *cleaned* sequence (``pose.smoothing.clean_sequence``) with ``bowling_arm`` set,
plus the bowler's standing height in pixels (``geometry.body.estimate_height_px``) so speed
thresholds don't depend on camera distance.

Events, in the order they are found:
- release: the frame of peak bowling-wrist speed while the wrist is above the shoulder,
  refined to the wrist-height peak within +-``release_refine_frames``.
- arm_horizontal: the last frame before release where the upper arm crosses horizontal going
  up.
- ffc / bfc: the front / back ankle comes down and stays still (vertical speed below
  ``contact_speed_h_per_s``) near its lowest point for ``contact_min_frames``, searching back
  from release / ffc.
- run_up_start / follow_through_end: hip horizontal speed crosses a fraction of its peak.
- bound: the highest hip point between run-up start and bfc.

If the events come out of order, confidence drops to ``low_confidence`` and the best guess is
still returned. From front or behind views the run-up moves along the camera axis, so hip
horizontal speed carries no information. There, run-up start and follow-through end fall back
to the clip edges and confidence is capped at ``no_horizontal_motion_confidence``.
"""

from __future__ import annotations

import logging

import numpy as np

from bowlsense.config import PhaseSettings
from bowlsense.errors import InvalidInputError, PhaseDetectionError
from bowlsense.geometry.angles import (
    FloatArray,
    elevation_angle,
    keypoint_track,
    linear_speed,
    midpoint,
)
from bowlsense.geometry.body import joint
from bowlsense.pose.smoothing import savgol_derivative
from bowlsense.schemas import PhaseEvents, PoseSequence

log = logging.getLogger(__name__)

HORIZONTAL_MOTION_VIEWS = frozenset({"side", "unknown"})


def _clamp(f: int, n: int) -> int:
    return int(min(max(f, 0), n - 1))


def find_release(wrist: FloatArray, shoulder: FloatArray, fps: float, refine_frames: int) -> int:
    """Peak wrist speed while the wrist is above the shoulder, refined to the wrist-height
    peak (smallest image y) nearby.

    Raises:
        PhaseDetectionError: if the wrist is never above the shoulder.
    """
    speed = linear_speed(wrist, fps)
    above = wrist[:, 1] < shoulder[:, 1]
    candidates = np.flatnonzero(above & np.isfinite(speed))
    if candidates.size == 0:
        raise PhaseDetectionError(
            "The bowling wrist is never above the shoulder, so no release was found. Check "
            "that the clip contains a delivery and the bowling arm is visible."
        )
    r0 = int(candidates[np.argmax(speed[candidates])])
    n = wrist.shape[0]
    lo, hi = _clamp(r0 - refine_frames, n), _clamp(r0 + refine_frames, n)
    heights = wrist[lo : hi + 1, 1]
    if not np.isfinite(heights).any():
        return r0
    return lo + int(np.nanargmin(heights))


def find_arm_horizontal(shoulder: FloatArray, elbow: FloatArray, release: int) -> int | None:
    """The last frame before release where the upper arm crosses horizontal going up."""
    elev = elevation_angle(shoulder, elbow)
    for f in range(release - 1, 0, -1):
        if elev[f - 1] < 0.0 <= elev[f]:
            return f
    return None


def find_contact(
    ankle: FloatArray,
    lo: int,
    hi: int,
    fps: float,
    height_px: float,
    settings: PhaseSettings,
    raw_ankle: FloatArray | None = None,
) -> int | None:
    """The latest frame in ``[lo, hi]`` where the ankle lands, searching backward from ``hi``.

    1. On the smoothed track, find where the ankle comes down and then stays still (vertical
       speed below ``contact_speed_h_per_s``) near its lowest point for
       ``contact_min_frames`` frames.
    2. Smoothing spreads the sudden stop of a foot strike over several frames, so the impact
       itself is pinned on the *unsmoothed* track (``raw_ankle``) with a change-point fit
       (``fit_impact_time``) in the window just before the still period.

    Returns ``None`` if there is no such landing in the window.
    """
    n = ankle.shape[0]
    y = ankle[:, 1]
    vy = savgol_derivative(y, fps, settings.contact_velocity_window_s)
    if not np.isfinite(y[lo : hi + 1]).any():
        return None
    ground = float(np.nanmax(y[lo : hi + 1]))  # image y grows downward
    near = y >= ground - settings.contact_ground_tol_pct / 100.0 * height_px
    still = (np.abs(vy) < settings.contact_speed_h_per_s * height_px) & near
    still &= np.isfinite(vy)
    m = settings.contact_min_frames
    for f in range(hi, lo - 1, -1):
        if f + m > n:
            continue
        if still[f : f + m].all() and (f == 0 or not still[f - 1]):
            return _impact_frame(
                raw_ankle if raw_ankle is not None else ankle, f, lo, fps, settings
            )
    return None


def fit_impact_time(t: FloatArray, y: FloatArray, lo_t: float, hi_t: float, order: int) -> float:
    """Change-point fit of a foot strike: least-squares fit of "descending curve, then flat"
    to ``y(t)``, searching the impact time over ``[lo_t, hi_t]`` in 0.1-frame steps.

    For a candidate impact ``tau`` the model is linear in its parameters::

        y = ground + v * min(t - tau, 0) + a * min(t - tau, 0)**2   (a only if order == 2)

    so each candidate is an ordinary least-squares solve, and the best ``tau`` minimises the
    residual. Every sample on both sides of the impact contributes, which averages out noise.
    """
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    best_tau, best_sse = lo_t, np.inf
    for tau in np.arange(lo_t, hi_t + 1e-9, 0.1):
        d = np.minimum(t - tau, 0.0)
        cols = [np.ones_like(t), d] + ([d**2] if order == 2 else [])
        basis = np.stack(cols, axis=1)
        if basis.shape[0] <= basis.shape[1]:
            continue
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        # Before impact d < 0 and the ankle is above ground (smaller image y), so a foot
        # coming down needs a positive coefficient.
        if coef[1] <= 0:
            continue
        sse = float(np.sum((basis @ coef - y) ** 2))
        if sse < best_sse:
            best_tau, best_sse = float(tau), sse
    return best_tau


def _impact_frame(
    ankle: FloatArray,
    still_onset: int,
    lo: int,
    fps: float,
    settings: PhaseSettings,
) -> int:
    """Pin the foot strike on the (unsmoothed) ankle track with a change-point fit.

    The smoothed track's still period starts a few frames after the real impact, so the
    impact is searched for within one smoothing window before it. The contact frame is the
    first frame at or after the fitted impact time.
    """
    n = ankle.shape[0]
    w = round(settings.contact_velocity_window_s * fps)
    k = max(3, round(settings.contact_fit_s * fps))
    search_lo = max(lo, still_onset - w)
    t0, t1 = max(0, search_lo - k), min(n, still_onset + k + 1)
    frames = np.arange(t0, t1, dtype=np.float64)
    tau = fit_impact_time(
        frames,
        ankle[t0:t1, 1],
        float(search_lo - 1),
        float(still_onset),
        settings.contact_fit_order,
    )
    return int(min(max(np.ceil(tau - 1e-9), search_lo), still_onset))


def first_sustained(mask: object, hold: int, start: int = 0) -> int | None:
    """First index ``>= start`` where ``mask`` is True for ``hold`` consecutive frames
    (fewer at the very end of the clip)."""
    m = np.asarray(mask, dtype=bool)
    n = m.size
    for f in range(start, n):
        if m[f : min(f + hold, n)].all():
            return f
    return None


def hip_speed(seq: PoseSequence, smooth_s: float) -> FloatArray:
    """Horizontal hip speed (px/s). Hip translation is a slow signal, so it gets its own,
    longer Savitzky-Golay derivative to keep noise from moving the threshold crossings.
    """
    hips_x = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))[:, 0]
    return np.abs(savgol_derivative(hips_x, seq.fps, smooth_s))


def detect_phases(
    seq: PoseSequence,
    height_px: float,
    settings: PhaseSettings,
    raw: PoseSequence | None = None,
    release: int | None = None,
) -> PhaseEvents:
    """Detect delivery events in a cleaned sequence.

    ``raw`` is the same clip gated and gap-filled but *not* smoothed
    (``clean_sequence(..., smooth=False)``). When given, foot-strike frames are pinned on
    it, which is more accurate; without it the smoothed track is used.

    ``release`` overrides the rule-based release (e.g. from the learned phase model); every
    other event is then found relative to it.

    Raises:
        InvalidInputError: if the sequence is empty or ``height_px`` is not positive.
        PhaseDetectionError: if no release can be found at all.
    """
    n = len(seq.frames)
    if n < 2:
        raise InvalidInputError("Phase detection needs at least two frames.")
    if not np.isfinite(height_px) or height_px <= 0:
        raise InvalidInputError(f"height_px must be positive, got {height_px}.")
    fps, arm = seq.fps, seq.bowling_arm
    notes: list[str] = []

    def track(role: str, part: str) -> FloatArray:
        return keypoint_track(seq, joint(arm, role, part))  # type: ignore[arg-type]

    shoulder, elbow, wrist = (
        track("bowling", "shoulder"),
        track("bowling", "elbow"),
        track("bowling", "wrist"),
    )
    if release is None:
        release = find_release(wrist, shoulder, fps, settings.release_refine_frames)
    elif not 0 <= release < n:
        raise InvalidInputError(f"release {release} is outside the {n}-frame sequence.")

    arm_h = find_arm_horizontal(shoulder, elbow, release)
    if arm_h is None:
        lo = _clamp(release - round(0.3 * fps), n)
        elev = np.abs(elevation_angle(shoulder[lo:release], elbow[lo:release]))
        arm_h = lo + int(np.nanargmin(elev)) if np.isfinite(elev).any() else max(release - 1, 0)
        notes.append("arm never crossed horizontal before release; used the closest frame")

    if raw is not None and len(raw.frames) != n:
        raise InvalidInputError("raw and seq must have the same number of frames.")

    def contact(role: str, lo: int, hi: int, label: str) -> int:
        ankle = track(role, "ankle")
        name = joint(arm, role, "ankle")  # type: ignore[arg-type]
        raw_ankle = keypoint_track(raw, name) if raw is not None else None
        found = find_contact(ankle, lo, hi, fps, height_px, settings, raw_ankle)
        if found is not None:
            return found
        notes.append(f"no clear {label} landing; used the ankle's lowest point")
        seg = ankle[lo : hi + 1, 1]
        return lo + int(np.nanargmax(seg)) if np.isfinite(seg).any() else lo

    ffc = contact("front", _clamp(release - round(settings.ffc_window_s * fps), n), release, "ffc")
    bfc = contact(
        "back",
        _clamp(ffc - round(settings.bfc_window_s * fps), n),
        max(ffc - 1, 0),
        "bfc",
    )

    hip_spd = hip_speed(seq, settings.hip_speed_smooth_s)
    horizontal = seq.view in HORIZONTAL_MOTION_VIEWS and np.isfinite(hip_spd).any()
    if horizontal:
        peak = float(np.nanmax(hip_spd))
        hold = max(1, round(settings.speed_hold_s * fps))
        start = first_sustained(hip_spd > settings.run_up_speed_frac * peak, hold)
        run_up_start = 0 if start is None else start
        end = first_sustained(
            hip_spd < settings.follow_through_speed_frac * peak, hold, release + 1
        )
        follow_end = n - 1 if end is None else end
    else:
        run_up_start, follow_end = 0, n - 1
        notes.append(f"{seq.view} view: run-up start and follow-through end set to clip edges")

    hips_y = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))[:, 1]
    seg = hips_y[run_up_start : bfc + 1]
    bound = run_up_start + int(np.nanargmin(seg)) if np.isfinite(seg).any() else run_up_start

    events = PhaseEvents(
        run_up_start=run_up_start,
        bound=bound,
        bfc=bfc,
        ffc=ffc,
        arm_horizontal=arm_h,
        release=release,
        follow_through_end=follow_end,
        confidence=1.0,
    )
    confidence = 1.0 if horizontal else settings.no_horizontal_motion_confidence
    violations = events.ordering_violations()
    if violations:
        notes.append("events out of order: " + "; ".join(violations))
    if violations or any("used the" in note for note in notes):
        confidence = min(confidence, settings.low_confidence)
    for note in notes:
        log.warning("phase detection: %s", note)
    return events.model_copy(update={"confidence": confidence})
