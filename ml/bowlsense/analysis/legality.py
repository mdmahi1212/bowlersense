"""Module A: elbow-extension legality screening and front-foot no-ball check.

2D video screening only. The official ICC test uses 3D motion capture; ``LegalityResult.note``
always says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from bowlsense.config import LegalitySettings
from bowlsense.errors import CalibrationError, InvalidInputError
from bowlsense.geometry.angles import FloatArray, joint_angle, keypoint_array, keypoint_track
from bowlsense.geometry.body import joint
from bowlsense.schemas import KEYPOINT_NAMES, Confidence, LegalityResult, PhaseEvents, PoseSequence

Point = tuple[float, float]


def _track3(seq: PoseSequence, name: str) -> FloatArray:
    """(frames, 3) x, y, z of one joint; ``NaN`` where it or its depth is missing."""
    out = np.full((len(seq.frames), 3), np.nan)
    for i, f in enumerate(seq.frames):
        k = f.keypoints.get(name)
        if k is not None and k.z is not None:
            out[i] = (k.x, k.y, k.z)
    return out


ARM_3D_MIN_COVERAGE = 0.8  # use 3D only when the lifted depth covers most frames


def arm_is_3d(seq: PoseSequence) -> bool:
    """True when the bowling arm has 3D-lifted depth (MotionBERT) on most frames."""
    if not seq.frames:
        return False
    names = [joint(seq.bowling_arm, "bowling", p) for p in ("shoulder", "elbow", "wrist")]
    have = np.all([np.isfinite(_track3(seq, n)[:, 2]) for n in names], axis=0)
    return float(np.mean(have)) >= ARM_3D_MIN_COVERAGE


def elbow_angles(seq: PoseSequence) -> FloatArray:
    """Bowling-arm elbow angle per frame (180 = straight); ``NaN`` where a joint is missing.

    Measured in 3D when the sequence has MotionBERT depth (``pose.lift_3d``), which removes
    the foreshortening a 2D camera sees when the arm swings towards or away from it."""
    arm = seq.bowling_arm
    names = [joint(arm, "bowling", p) for p in ("shoulder", "elbow", "wrist")]
    if arm_is_3d(seq):
        return joint_angle(*(_track3(seq, n) for n in names))
    return joint_angle(*(keypoint_track(seq, n) for n in names))


def elbow_extension(angles: FloatArray, arm_horizontal: int, release: int) -> float:
    """How much the arm straightens between arm-horizontal and release.

    ``max(angle over [AH, release]) - min(angle over [AH, frame of that max])``: from the
    most-flexed point up to the straightest point that follows it.

    Raises:
        InvalidInputError: if the window is empty or has no valid angle.
    """
    if not 0 <= arm_horizontal <= release < angles.shape[0]:
        raise InvalidInputError(
            f"Need 0 <= arm_horizontal ({arm_horizontal}) <= release ({release}) < "
            f"{angles.shape[0]} frames."
        )
    window = angles[arm_horizontal : release + 1]
    if not np.isfinite(window).any():
        raise InvalidInputError(
            "The elbow is not visible between arm-horizontal and release, so elbow extension "
            "can't be measured. Film side-on with the bowling arm unobstructed."
        )
    k = int(np.nanargmax(window))
    return float(window[k] - np.nanmin(window[: k + 1]))


def fitted_elbow_angles(
    angles: FloatArray, arm_horizontal: int, release: int, fps: float, settings: LegalitySettings
) -> FloatArray:
    """Smooth elbow angles over ``[AH, release]`` with a least-squares polynomial.

    The elbow straightens in about 50 ms (a handful of frames), so general keypoint
    smoothing flattens it, while raw angles are noisy and ``max - min`` picks up that noise.
    A low-order polynomial fitted over the window (padded by ``fit_pad_s``) follows a smooth
    straightening while averaging noise across every frame. Outside the window the result is
    ``NaN``.

    Raises:
        InvalidInputError: if too few valid angles are available for the fit.
    """
    n = angles.shape[0]
    pad = round(settings.fit_pad_s * fps)
    lo, hi = max(0, arm_horizontal - pad), min(n - 1, release + pad)
    t = np.arange(lo, hi + 1, dtype=np.float64)
    seg = angles[lo : hi + 1]
    ok = np.isfinite(seg)
    if int(ok.sum()) <= settings.fit_degree:
        raise InvalidInputError(
            "Too few frames with a visible elbow around release to measure extension. "
            "Film at 120 fps or more with the bowling arm unobstructed."
        )
    coeffs = np.polyfit(t[ok] - arm_horizontal, seg[ok], settings.fit_degree)
    out = np.full(n, np.nan)
    window = np.arange(arm_horizontal, release + 1, dtype=np.float64)
    out[arm_horizontal : release + 1] = np.clip(
        np.polyval(coeffs, window - arm_horizontal), 0.0, 180.0
    )
    return out


def _confidence(seq: PoseSequence, phases: PhaseEvents, settings: LegalitySettings) -> Confidence:
    _, conf = keypoint_array(seq)
    names = [joint(seq.bowling_arm, "bowling", p) for p in ("elbow", "wrist")]
    cols = [KEYPOINT_NAMES.index(n) for n in names]
    window = conf[phases.arm_horizontal : phases.release + 1][:, cols]
    mean_conf = float(window.mean()) if window.size else 0.0
    if seq.fps >= settings.high_conf_fps and mean_conf > settings.high_conf_keypoint:
        return "high"
    if seq.fps >= settings.medium_conf_fps:
        return "medium"
    return "low"


def front_foot_no_ball(
    seq: PoseSequence, phases: PhaseEvents, popping_crease: Sequence[Point]
) -> bool:
    """True if no part of the front foot is behind the popping crease at front-foot contact.

    The front heel is used (the toe if the heel isn't detected). "Behind" is the bowler's side
    of the line, taken from where the back foot landed at back-foot contact. A heel exactly
    on the line is not behind it (no-ball).

    Raises:
        CalibrationError: if the crease isn't two distinct points or the back foot sits on it.
        InvalidInputError: if the needed keypoints are missing.
    """
    if len(popping_crease) != 2:
        raise CalibrationError("The popping crease needs exactly two image points.")
    a = np.asarray(popping_crease[0], dtype=np.float64)
    b = np.asarray(popping_crease[1], dtype=np.float64)
    direction = b - a
    if float(np.hypot(*direction)) < 1e-9:
        raise CalibrationError("The two popping-crease points are identical.")

    def side(p: FloatArray) -> float:
        rel = p - a
        return float(direction[0] * rel[1] - direction[1] * rel[0])

    arm = seq.bowling_arm
    back = keypoint_track(seq, joint(arm, "back", "ankle"))[phases.bfc]
    if not np.isfinite(back).all():
        raise InvalidInputError("Back ankle not visible at back-foot contact.")
    bowler_side = side(back)
    if bowler_side == 0.0:
        raise CalibrationError("The back foot is on the crease line; check the crease points.")
    heel = keypoint_track(seq, joint(arm, "front", "heel"))[phases.ffc]
    if not np.isfinite(heel).all():
        heel = keypoint_track(seq, joint(arm, "front", "toe"))[phases.ffc]
    if not np.isfinite(heel).all():
        raise InvalidInputError("Front heel and toe not visible at front-foot contact.")
    behind = side(heel) * bowler_side > 0.0
    return not behind


def _arm_horizontal_plausible(
    seq: PoseSequence, phases: PhaseEvents, settings: LegalitySettings
) -> bool:
    """AH must fall inside the delivery stride (after back-foot contact, before release)
    and no more than ``screen_max_window_s`` before release."""
    window_s = (phases.release - phases.arm_horizontal) / seq.fps
    return phases.bfc <= phases.arm_horizontal < phases.release and (
        window_s <= settings.screen_max_window_s
    )


def screening_problems(
    seq: PoseSequence, phases: PhaseEvents, extension: float, settings: LegalitySettings
) -> list[str]:
    """Why this clip can't be screened for elbow extension (empty list = it can)."""
    problems = []
    if seq.fps < settings.screen_min_fps:
        problems.append(
            f"the video is {seq.fps:.0f} fps; at least {settings.screen_min_fps:.0f} fps is "
            "needed to see the elbow straighten"
        )
    if seq.view not in settings.screen_views:
        problems.append(f"the camera is {seq.view}-on; elbow angles need a side-on camera")
    if not phases.bfc < phases.release or phases.release <= 0:
        problems.append("the release couldn't be placed after the delivery stride")
    elif phases.confidence < settings.screen_min_phase_confidence and not (
        phases.bfc <= phases.ffc <= phases.release
    ):
        problems.append("the delivery phases couldn't be found reliably")
    if extension > settings.screen_max_plausible_deg:
        problems.append(
            f"the measured {extension:.0f} deg is beyond what an elbow can straighten, so the "
            "arm wasn't tracked correctly"
        )
    return problems


def legality_verdict(
    extension: float, limit: float, flagged: bool, problems: Sequence[str] = ()
) -> tuple[Literal["legal", "not_legal", "cannot_judge"], str]:
    """The plain-language verdict for one delivery (2D screening, not the ICC test).

    The verdict always follows the measured angle (<= limit legal, > limit not legal). When
    the clip can't support a trustworthy measurement, the text says the verdict is unverified
    and why."""
    if flagged:
        text = (
            f"Not legal: the elbow straightened {extension:.1f} deg, more than the {limit:.0f} deg "
            "limit."
        )
    else:
        text = (
            f"Legal: the elbow straightened {extension:.1f} deg, within the {limit:.0f} deg limit."
        )
    if problems:
        text += (
            " Unverified: " + "; ".join(problems) + ". Treat this as a rough guide and re-film "
            "side-on at 120+ fps to confirm."
        )
    return ("not_legal" if flagged else "legal"), text


def assess_legality(
    seq: PoseSequence,
    phases: PhaseEvents,
    settings: LegalitySettings,
    raw: PoseSequence | None = None,
    popping_crease: Sequence[Point] | None = None,
) -> LegalityResult:
    """Screen the bowling action: elbow extension vs threshold, plus no-ball if calibrated.

    ``seq`` is the cleaned (smoothed) sequence. ``raw`` is the same clip gated and gap-filled
    but not smoothed (``clean_sequence(..., smooth=False)``); when given, elbow angles come
    from it, so general smoothing doesn't flatten the extension. Either way the angles are
    then fitted over the AH-release window (``fitted_elbow_angles``).

    Elbow angles only move in the image plane from a side-on camera, so other views cap
    confidence at "low". Clips that can't support a trustworthy measurement are reported as
    not screened: the verdict still follows the angle, but ``screened`` is False and the
    text says why it is unverified; see ``screening_problems``.
    """
    angles = elbow_angles(raw if raw is not None else seq)
    estimated_ah = not _arm_horizontal_plausible(seq, phases, settings)
    if estimated_ah:  # use a fixed window before release instead of an impossible AH
        ah = max(0, phases.release - max(2, round(settings.fallback_window_s * seq.fps)))
        phases = phases.model_copy(update={"arm_horizontal": ah})
    try:
        fitted = fitted_elbow_angles(
            angles, phases.arm_horizontal, phases.release, seq.fps, settings
        )
        extension = elbow_extension(fitted, phases.arm_horizontal, phases.release)
    except InvalidInputError as err:  # elbow hidden around release: report, don't abort
        return LegalityResult(
            elbow_extension_deg=0.0,
            threshold_deg=settings.elbow_extension_threshold_deg,
            flagged=False,
            screened=False,
            measured=False,
            verdict="cannot_judge",
            verdict_text=(
                "Can't measure: the bowling elbow isn't visible around release in this clip, so "
                "there is no angle to judge. Film side-on with the bowling arm in view."
            ),
            not_screened_reason=f"Not measured: {err}",
            confidence="low",
            note="2D video screening only. The official ICC test uses 3D motion capture.",
            front_foot_no_ball=None,
        )
    confidence = _confidence(seq, phases, settings)
    note = "2D video screening only. The official ICC test uses 3D motion capture."
    if arm_is_3d(raw if raw is not None else seq):
        note += " The elbow was measured in 3D estimated from the video (MotionBERT)."
    if estimated_ah:
        confidence = "low"
        note += (
            f" Arm-horizontal couldn't be found reliably, so the elbow was measured over the "
            f"last {settings.fallback_window_s * 1000:.0f} ms before release."
        )
    if seq.view not in ("side", "unknown"):
        confidence = "low"
        note += " Elbow angles are only reliable from a side-on camera."
    problems = screening_problems(seq, phases, extension, settings)
    reason = None
    if problems:
        reason = "Unverified: " + "; ".join(problems) + "."
        note += " " + reason
        confidence = "low"
    limit = settings.elbow_extension_threshold_deg
    flagged = extension > limit
    verdict, verdict_text = legality_verdict(extension, limit, flagged, problems)
    return LegalityResult(
        elbow_extension_deg=extension,
        threshold_deg=limit,
        flagged=flagged,
        screened=not problems,
        verdict=verdict,
        verdict_text=verdict_text,
        not_screened_reason=reason,
        confidence=confidence,
        note=note,
        front_foot_no_ball=(
            None if popping_crease is None else front_foot_no_ball(seq, phases, popping_crease)
        ),
    )
