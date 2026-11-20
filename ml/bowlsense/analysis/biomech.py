"""Biomechanics: a per-delivery table of body measurements and the joint curves behind them.

Angles are 2D (in the image plane), so each measurement says which camera view it needs; any
that can't be measured from this clip are listed with the reason instead of a number.
Distances are % of standing height and speeds are body heights per second, so they don't
depend on how far the camera is. Where a fault or injury module measures the same thing, the
same function is used, so the numbers always agree.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bowlsense.analysis import faults as fm
from bowlsense.analysis.injury import (
    arm_angular_velocity_deg_s,
    arm_slot_deg,
    trunk_rotation_speed_deg_s,
)
from bowlsense.analysis.legality import elbow_angles
from bowlsense.config import Settings
from bowlsense.errors import BowlSenseError
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
from bowlsense.schemas import (
    AlignmentResult,
    AngleSeries,
    BiomechMetric,
    PhaseEvents,
    PoseSequence,
)

SIDE = ("side", "unknown")
FRONT = ("front", "behind")


def knee_angles(seq: PoseSequence, role: str) -> FloatArray:
    """Knee angle per frame (180 = straight) of the front or back leg."""
    arm = seq.bowling_arm
    return joint_angle(
        keypoint_track(seq, joint(arm, role, "hip")),  # type: ignore[arg-type]
        keypoint_track(seq, joint(arm, role, "knee")),  # type: ignore[arg-type]
        keypoint_track(seq, joint(arm, role, "ankle")),  # type: ignore[arg-type]
    )


def trunk_lean_deg(seq: PoseSequence) -> FloatArray:
    """Trunk angle from vertical (hip midpoint -> shoulder midpoint), unsigned."""
    sh = midpoint(keypoint_track(seq, "l_shoulder"), keypoint_track(seq, "r_shoulder"))
    hp = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
    return np.asarray(
        np.degrees(np.arctan2(np.abs(sh[:, 0] - hp[:, 0]), hp[:, 1] - sh[:, 1])),
        dtype=np.float64,
    )


def upper_arm_elevation_deg(seq: PoseSequence) -> FloatArray:
    """Bowling upper arm (shoulder -> elbow) angle above horizontal (90 = straight up)."""
    arm = seq.bowling_arm
    sh = keypoint_track(seq, joint(arm, "bowling", "shoulder"))
    el = keypoint_track(seq, joint(arm, "bowling", "elbow"))
    return np.asarray(
        np.degrees(np.arctan2(sh[:, 1] - el[:, 1], np.abs(el[:, 0] - sh[:, 0]))),
        dtype=np.float64,
    )


def curves(seq: PoseSequence) -> dict[str, FloatArray]:
    """The joint-angle curves shown on the biomechanics tab (degrees per frame)."""
    return {
        "bowling_elbow": elbow_angles(seq),
        "front_knee": knee_angles(seq, "front"),
        "back_knee": knee_angles(seq, "back"),
        "trunk_lean": trunk_lean_deg(seq),
        "upper_arm_elevation": upper_arm_elevation_deg(seq),
    }


def biomech_curves(seq: PoseSequence, window_s: float) -> list[AngleSeries]:
    """Angle and angular-velocity series (``<name>`` and ``<name>_velocity``, deg/s)."""
    out = []
    frames = list(range(len(seq.frames)))
    for name, values in curves(seq).items():
        vel = savgol_derivative(values, seq.fps, window_s)
        for key, series in ((name, values), (f"{name}_velocity", vel)):
            out.append(
                AngleSeries(
                    name=key,
                    values=[float(v) if np.isfinite(v) else None for v in series],
                    frames=frames,
                )
            )
    return out


def _at(values: FloatArray, frame: int) -> float | None:
    if 0 <= frame < values.shape[0] and np.isfinite(values[frame]):
        return float(values[frame])
    return None


def biomechanics(
    seq: PoseSequence,
    phases: PhaseEvents,
    height_px: float,
    alignment: AlignmentResult,
    settings: Settings,
    raw: PoseSequence | None = None,
) -> list[BiomechMetric]:
    """Every biomechanical measurement of one delivery (``seq`` is the cleaned clip)."""
    out: list[BiomechMetric] = []
    view = seq.view
    fps = seq.fps
    ctx = fm.FaultContext(seq=seq, phases=phases, height_px=height_px, settings=settings, raw=raw)

    def add(
        key: str, name: str, group: str, unit: str, desc: str, views: tuple[str, ...],
        get: Callable[[], float | None],
    ) -> None:  # fmt: skip
        if view not in views:
            need = "a side-on camera" if views == SIDE else "a front-on or behind camera"
            out.append(BiomechMetric(key=key, name=name, group=group, value=None, unit=unit,
                                     description=desc, not_measured=f"needs {need}"))  # fmt: skip
            return
        try:
            value = get()
            reason = None if value is not None else "joints not visible at that moment"
        except BowlSenseError as exc:
            value, reason = None, str(exc)
        if value is not None and not np.isfinite(value):
            value, reason = None, "joints not visible at that moment"
        out.append(BiomechMetric(key=key, name=name, group=group, value=value, unit=unit,
                                 description=desc, not_measured=reason))  # fmt: skip

    c = curves(seq)
    any_view = SIDE + FRONT
    bfc, ffc, rel, ah = phases.bfc, phases.ffc, phases.release, phases.arm_horizontal

    # --- back-foot contact
    add("back_knee_bfc", "Back-knee angle", "Back-foot contact", "deg",
        "Back leg at back-foot contact (180 = straight).", SIDE,
        lambda: _at(c["back_knee"], bfc))  # fmt: skip
    add("trunk_lean_bfc", "Trunk lean", "Back-foot contact", "deg",
        "Trunk angle from vertical at back-foot contact.", any_view,
        lambda: _at(c["trunk_lean"], bfc))  # fmt: skip
    add("shoulder_alignment_bfc", "Shoulder alignment", "Back-foot contact", "deg",
        "Shoulder line turned from side-on (0) towards front-on (90).", FRONT,
        lambda: alignment.shoulder_angle_bfc_deg)  # fmt: skip
    add("hip_alignment_bfc", "Hip alignment", "Back-foot contact", "deg",
        "Hip line turned from side-on (0) towards front-on (90).", FRONT,
        lambda: alignment.hip_angle_bfc_deg)  # fmt: skip
    add("hip_shoulder_sep_bfc", "Hip-shoulder separation", "Back-foot contact", "deg",
        "Difference between hip and shoulder alignment (large = mixed action).", FRONT,
        lambda: alignment.hip_shoulder_diff_bfc_deg)  # fmt: skip

    # --- front-foot contact
    add("front_knee_ffc", "Front-knee angle", "Front-foot contact", "deg",
        "Front leg at front-foot contact (180 = straight).", SIDE,
        lambda: _at(c["front_knee"], ffc))  # fmt: skip
    add("upper_arm_ffc", "Bowling upper-arm elevation", "Front-foot contact", "deg",
        "Upper arm above horizontal at front-foot contact.", SIDE,
        lambda: _at(c["upper_arm_elevation"], ffc))  # fmt: skip
    add("front_arm_height_ffc", "Front-arm height", "Front-foot contact", "% of height",
        "Non-bowling wrist above (+) or below (-) its shoulder.", SIDE,
        lambda: -fm._weak_front_arm(ctx)[0])  # fmt: skip
    add("hip_shoulder_sep_ffc", "Hip-shoulder separation", "Front-foot contact", "deg",
        "Hips leading the shoulders at front-foot contact.", FRONT,
        lambda: alignment.hip_shoulder_sep_ffc_deg)  # fmt: skip
    add("counter_rotation", "Shoulder counter-rotation", "Front-foot contact", "deg",
        "How far the shoulders turn back towards side-on between BFC and FFC.", FRONT,
        lambda: alignment.counter_rotation_deg)  # fmt: skip

    # --- release
    add("front_knee_release", "Front-knee angle", "Release", "deg",
        "Front leg at release (180 = straight; braced is about 160+).", SIDE,
        lambda: _at(c["front_knee"], rel))  # fmt: skip
    add("knee_flexion_change", "Front-knee flexion FFC->release", "Release", "deg",
        "How much the front knee bends between contact and release (collapse).", SIDE,
        lambda: fm._knee_collapse(ctx)[0])  # fmt: skip
    add("elbow_release", "Bowling-elbow angle", "Release", "deg",
        "Bowling elbow at release (180 = straight).", SIDE,
        lambda: _at(c["bowling_elbow"], rel))  # fmt: skip
    add("trunk_lean_release", "Trunk lean", "Release", "deg",
        "Trunk angle from vertical at release.", any_view,
        lambda: _at(c["trunk_lean"], rel))  # fmt: skip
    add("lateral_flexion", "Trunk side-bend", "Release", "deg",
        "Side-bend away from the bowling arm at release.", FRONT,
        lambda: fm._lateral_flexion(ctx)[0])  # fmt: skip
    add(
        "release_height",
        "Release height",
        "Release",
        "% of height",
        "Bowling wrist height at release.",
        SIDE,
        lambda: fm.release_height_pct(ctx)[0],
    )
    add("arm_slot", "Arm slot", "Release", "deg from vertical",
        "Bowling arm angle from vertical at release (0 = straight up).", FRONT,
        lambda: arm_slot_deg(seq, phases))  # fmt: skip
    add("head_offset", "Head offset from front foot", "Release", "% of height",
        "Sideways distance of the head from the front foot at release.", FRONT,
        lambda: fm._head_fall(ctx)[0])  # fmt: skip

    # --- stride and run-up
    add(
        "stride",
        "Delivery stride",
        "Stride & run-up",
        "% of height",
        "Ankle-to-ankle distance at front-foot contact.",
        SIDE,
        lambda: fm.stride_pct(ctx)[0],
    )
    add("runup_decel", "Run-up speed loss", "Stride & run-up", "%",
        "Speed lost over the last strides before the crease.", SIDE,
        lambda: fm._runup_decel(ctx)[0])  # fmt: skip
    add("follow_through", "Follow-through", "Stride & run-up", "% of stride",
        "Hip travel after release relative to the stride.", SIDE,
        lambda: fm._short_follow(ctx)[0])  # fmt: skip

    # --- speeds
    def peak_hip_speed() -> float | None:
        sp = hip_speed(seq, settings.faults.runup_speed_smooth_s)[: max(1, bfc)]
        return float(np.nanmax(sp)) / height_px if np.isfinite(sp).any() else None

    def peak_wrist_speed() -> float | None:
        sp = linear_speed(keypoint_track(seq, joint(seq.bowling_arm, "bowling", "wrist")), fps)
        hi = min(len(sp) - 1, rel + round(0.05 * fps))
        seg = sp[max(0, bfc) : hi + 1]
        return float(np.nanmax(seg)) / height_px if np.isfinite(seg).any() else None

    def knee_extension_rate() -> float | None:
        vel = savgol_derivative(c["front_knee"], fps, settings.pose.savgol_window_s)
        seg = vel[ffc : rel + 1]
        return float(np.nanmax(seg)) if np.isfinite(seg).any() else None

    add("runup_speed", "Peak run-up speed", "Speeds", "body heights/s",
        "Fastest hip speed in the run-up (x your height in m = m/s).", SIDE,
        peak_hip_speed)  # fmt: skip
    add("wrist_speed", "Peak bowling-wrist speed", "Speeds", "body heights/s",
        "Fastest bowling-hand speed around release.", SIDE, peak_wrist_speed)  # fmt: skip
    add("arm_angular_velocity", "Peak arm angular velocity", "Speeds", "deg/s",
        "Fastest rotation of the bowling upper arm (about 1500-2000 for pace).", SIDE,
        lambda: arm_angular_velocity_deg_s(seq, phases, settings.pose.savgol_window_s))  # fmt: skip
    add("trunk_rotation_speed", "Peak trunk rotation speed", "Speeds", "deg/s",
        "Fastest shoulder-line rotation between BFC and release.", FRONT,
        lambda: trunk_rotation_speed_deg_s(seq, phases, settings))  # fmt: skip
    add("knee_extension_rate", "Front-knee extension rate", "Speeds", "deg/s",
        "Peak rate of change of the front knee from FFC to release (+ straightening, - "
        "bending).", SIDE,
        knee_extension_rate)  # fmt: skip

    # --- timing
    add(
        "bfc_to_ffc",
        "Back-foot to front-foot contact",
        "Timing",
        "s",
        "Duration of the delivery stride.",
        any_view,
        lambda: (ffc - bfc) / fps,
    )
    add(
        "ffc_to_release",
        "Front-foot contact to release",
        "Timing",
        "s",
        "Time from front-foot landing to release.",
        any_view,
        lambda: (rel - ffc) / fps,
    )
    add(
        "ah_to_release",
        "Arm horizontal to release",
        "Timing",
        "s",
        "The window the elbow-legality check measures.",
        any_view,
        lambda: (rel - ah) / fps,
    )

    def chain(i: int) -> Callable[[], float]:
        return lambda: (rel - fm._chain_order(ctx)[1][i]) / fps

    for i, (part, label) in enumerate((("pelvis", "Pelvis"), ("trunk", "Trunk"), ("arm", "Arm"))):
        add(f"chain_{part}", f"{label} peak speed before release", "Kinetic chain", "s",
            "Ideal order: pelvis, then trunk, then arm (largest to smallest time).", SIDE,
            chain(i))  # fmt: skip
    add("chain_out_of_order", "Kinetic-chain sequencing error", "Kinetic chain", "s",
        "How far the pelvis-trunk-arm order is broken (0 = ideal).", SIDE,
        lambda: fm._chain_order(ctx)[0])  # fmt: skip
    return out
