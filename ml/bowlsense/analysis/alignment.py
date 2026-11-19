"""Module B: shoulder and hip alignment (action type, counter-rotation, separation).

Alignment is rotation about the vertical axis ("yaw"): 0 = side-on, 90 = front-on, negative
= turned past side-on. From a front or behind camera that rotation doesn't tilt the
shoulder line in the image, it narrows it, so yaw is recovered from the *signed apparent
width*::

    yaw = arcsin(signed width / full width)

The full width is calibrated in the run-up, when the chest faces the camera. Left/right
order in the image gives the sign, so turning past side-on comes out negative. Each width is
divided by the trunk length in the same frame, so the bowler getting bigger while running
towards the camera (perspective) cancels out.

From a side-on camera these angles can't be measured; every angle is ``None`` and the notes
say "needs front view".
"""

from __future__ import annotations

import numpy as np

from bowlsense.config import AlignmentSettings
from bowlsense.errors import PoseQualityError
from bowlsense.geometry.angles import FloatArray, keypoint_track, midpoint
from bowlsense.geometry.body import joint
from bowlsense.schemas import ActionType, AlignmentResult, PhaseEvents, PoseSequence

NEEDS_FRONT_VIEW = "needs front view"
FRONTAL_VIEWS = frozenset({"front", "behind"})


def _orientation_sign(seq: PoseSequence) -> float:
    """Sign that makes ``sign * (bowling-side x - front-side x)`` grow towards front-on.

    A right-arm bowler seen from the front (facing the camera) shows the bowling (right)
    shoulder on the image *left*, so the raw difference is negative when front-on: sign -1.
    Left-arm bowlers and the behind view each flip it.
    """
    arm = -1.0 if seq.bowling_arm == "right" else 1.0
    view = 1.0 if seq.view == "front" else -1.0
    return arm * view


def _normalised_width(seq: PoseSequence, part: str) -> FloatArray:
    """Signed width of a body line (shoulders or hips) / trunk length, per frame."""
    arm = seq.bowling_arm
    bowl = keypoint_track(seq, joint(arm, "bowling", part))
    front = keypoint_track(seq, joint(arm, "front", part))
    trunk = np.linalg.norm(
        midpoint(keypoint_track(seq, "l_shoulder"), keypoint_track(seq, "r_shoulder"))
        - midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip")),
        axis=1,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = _orientation_sign(seq) * (bowl[:, 0] - front[:, 0]) / trunk
    return np.asarray(np.where(trunk > 1e-9, out, np.nan), dtype=np.float64)


def _full_width(width: FloatArray, lo: int, hi: int, top_frac: float, label: str) -> float:
    seg = np.abs(width[lo:hi])
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        raise PoseQualityError(
            f"The {label} are not visible during the run-up, so alignment can't be "
            "calibrated. Keep the whole body in frame from the start of the run-up."
        )
    k = max(1, int(np.ceil(top_frac * seg.size)))
    full = float(np.median(np.sort(seg)[-k:]))
    if full <= 0:
        raise PoseQualityError(f"The {label} have zero width in the run-up.")
    return full


def full_width(seq: PoseSequence, phases: PhaseEvents, part: str, top_frac: float) -> float:
    """Full (front-on) normalised width of the shoulder or hip line, from the run-up.

    Use a smoothed sequence: on noisy data the widest frames are biased wide by the noise.
    """
    label = "shoulders" if part == "shoulder" else "hips"
    width = _normalised_width(seq, part)
    return _full_width(width, phases.run_up_start, phases.bound, top_frac, label)


def yaw_series(seq: PoseSequence, part: str, full: float) -> FloatArray:
    """Yaw (deg) of the shoulder or hip line per frame from a front/behind view, given the
    full width from ``full_width``."""
    width = _normalised_width(seq, part)
    return np.asarray(np.degrees(np.arcsin(np.clip(width / full, -1.0, 1.0))), dtype=np.float64)


def fitted_minimum(values: FloatArray, lo: int, hi: int, half: int) -> float:
    """Minimum of ``values[lo:hi+1]``, refined by a quadratic fit over ``+-half`` frames
    around the lowest sample (noise-averaged instead of picking the lowest noisy frame).
    Falls back to the plain minimum when the fit isn't a proper dip."""
    seg = values[lo : hi + 1]
    if not np.isfinite(seg).any():
        return float("nan")
    k = lo + int(np.nanargmin(seg))
    plain = float(values[k])
    a, b = max(0, k - half), min(values.shape[0] - 1, k + half)
    t = np.arange(a, b + 1, dtype=np.float64)
    v = values[a : b + 1]
    ok = np.isfinite(v)
    if int(ok.sum()) < 4:
        return plain
    c2, c1, c0 = np.polyfit(t[ok] - k, v[ok], 2)
    if c2 <= 0:
        return plain
    vertex = -c1 / (2.0 * c2)
    if abs(vertex) > half:
        return plain
    return float(c0 - c1 * c1 / (4.0 * c2))


def classify_action(shoulder_yaw: float, hip_yaw: float, settings: AlignmentSettings) -> ActionType:
    """Action type at back-foot contact from shoulder and hip yaw."""
    if abs(shoulder_yaw - hip_yaw) > settings.mixed_threshold_deg:
        return "mixed"
    if shoulder_yaw <= settings.closed_max_deg and hip_yaw <= settings.closed_max_deg:
        return "side_on"
    if shoulder_yaw >= settings.open_min_deg and hip_yaw >= settings.open_min_deg:
        return "front_on"
    return "semi_open"


def assess_alignment(
    seq: PoseSequence,
    phases: PhaseEvents,
    settings: AlignmentSettings,
    raw: PoseSequence | None = None,
) -> AlignmentResult:
    """Action type at BFC, shoulder counter-rotation (BFC to its minimum before FFC), and
    hip-shoulder separation at FFC.

    Counter-rotation is a sharp dip lasting well under 0.1 s, which keypoint smoothing fills
    in. When ``raw`` (the same clip gated and gap-filled but not smoothed) is given, the dip
    is measured on it, with a quadratic fit around the minimum to average out noise.

    Raises:
        PoseQualityError: if shoulders or hips can't be calibrated or are missing at BFC/FFC.
    """
    if seq.view not in FRONTAL_VIEWS:
        return AlignmentResult(
            action_type="unknown",
            shoulder_angle_bfc_deg=None,
            hip_angle_bfc_deg=None,
            hip_shoulder_diff_bfc_deg=None,
            counter_rotation_deg=None,
            counter_rotation_flagged=None,
            hip_shoulder_sep_ffc_deg=None,
            notes=[
                f"action type, counter-rotation and hip-shoulder separation: {NEEDS_FRONT_VIEW}"
                f" (camera view is '{seq.view}')"
            ],
        )
    top = settings.width_calib_top_frac
    sh_full = full_width(seq, phases, "shoulder", top)
    sh = yaw_series(seq, "shoulder", sh_full)
    hip = yaw_series(seq, "hip", full_width(seq, phases, "hip", top))
    for name, series, f in (
        ("shoulders", sh, phases.bfc),
        ("hips", hip, phases.bfc),
        ("shoulders", sh, phases.ffc),
        ("hips", hip, phases.ffc),
    ):
        if not np.isfinite(series[f]):
            raise PoseQualityError(f"The {name} are not visible at frame {f}.")

    sh_bfc, hip_bfc = float(sh[phases.bfc]), float(hip[phases.bfc])
    dip = yaw_series(raw, "shoulder", sh_full) if raw is not None else sh
    half = max(2, round(settings.min_fit_half_s * seq.fps))
    lowest = fitted_minimum(dip, phases.bfc, phases.ffc, half)
    counter = max(0.0, sh_bfc - min(lowest, sh_bfc))
    return AlignmentResult(
        action_type=classify_action(sh_bfc, hip_bfc, settings),
        shoulder_angle_bfc_deg=sh_bfc,
        hip_angle_bfc_deg=hip_bfc,
        hip_shoulder_diff_bfc_deg=hip_bfc - sh_bfc,
        counter_rotation_deg=counter,
        counter_rotation_flagged=counter > settings.counter_rotation_flag_deg,
        hip_shoulder_sep_ffc_deg=float(hip[phases.ffc] - sh[phases.ffc]),
        notes=[],
    )
