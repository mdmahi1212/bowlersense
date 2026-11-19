"""Synthetic bowling skeleton generator (spec Section 10).

Builds a 3D stick figure with anthropometric segment lengths, drives it through a keyframed
bowling action (run-up, bound, back-foot contact, front-foot contact, arm circle, release,
follow-through) and projects it orthographically to a side, front or behind camera view.

Every parameter a later module measures is set here, so tests can check the module gets it
back. Ground-truth phase frames are reported alongside the sequence.

World frame (units of standing height H = 1): +X is the run direction (towards the batter),
+Y is up, +Z is the bowler's left when facing +X. The action is built for a right-arm bowler
and mirrored (Z -> -Z, left/right names swapped) for a left-arm bowler.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from bowlsense.schemas import (
    KEYPOINT_NAMES,
    SKELETON_EDGES,
    Arm,
    FramePose,
    Keypoint,
    PhaseEvents,
    PoseSequence,
)

FloatArray = NDArray[np.float64]

# --- anthropometry (fractions of standing height; Winter / Drillis-Contini) -----------------
ANKLE_H = 0.039
SHANK = 0.246
THIGH = 0.245
HIP_JOINT_H = ANKLE_H + SHANK + THIGH  # 0.530
TRUNK = 0.288  # hip-joint centre to shoulder centre
HIP_WIDTH = 0.11  # between hip joints
SHOULDER_WIDTH = 0.23
UPPER_ARM = 0.186
FOREARM = 0.146
HAND = 0.108  # wrist to fingertips; the ball sits roughly a hand-length past the wrist
NECK = 0.052
HEAD_CENTRE = 0.066  # neck top to head centre
HEEL_BACK, HEEL_DOWN = 0.03, 0.03
TOE_FWD, TOE_DOWN = 0.12, 0.035

# --- timeline (seconds) ---------------------------------------------------------------------
T_STAND = 0.25  # standing still before the run-up
STEP_PERIOD = 0.25
CONTACT = 0.11  # foot ground-contact time per run-up step
N_RUNUP_STEPS = 8  # 4 strides; the "final 3 strides" are the last 6 steps
BOUND_RISE = 0.13  # take-off to highest point
BOUND_FALL = 0.13  # highest point to back-foot contact
BFC_TO_FFC = 0.13
FFC_TO_RELEASE = 0.10
AH_BEFORE_RELEASE = 0.05
AFTER_RELEASE = 1.0

V_PEAK = 3.2  # run-up speed, heights per second (~5.8 m/s for a 1.8 m bowler)
BASE_ELBOW = 175.0  # elbow angle with the arm "straight" at release
FRONT_KNEE_AT_FFC = 172.0
FRONT_ELBOW = 165.0

ACTION_YAWS: dict[str, tuple[float, float]] = {
    # (hip yaw, shoulder yaw) at back-foot contact; 0 deg = side-on, 90 deg = front-on.
    "side_on": (15.0, 10.0),
    "semi_open": (40.0, 40.0),
    "front_on": (70.0, 65.0),
    "mixed": (65.0, 15.0),
}

ActionType = Literal["side_on", "semi_open", "front_on", "mixed"]
SynthView = Literal["side", "front", "behind"]


class SynthParams(BaseModel):
    """Controllable parameters of a synthetic delivery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fps: float = Field(default=120.0, gt=0, le=1000)
    height_px: float = Field(default=300.0, ge=50, le=2000)
    bowling_arm: Arm = "right"
    view: SynthView = "side"
    action_type: ActionType = "semi_open"
    elbow_extension_deg: float = Field(default=8.0, ge=0, le=40)
    counter_rotation_deg: float = Field(default=10.0, ge=0, le=60)
    knee_flex_change_deg: float = Field(default=8.0, ge=0, le=50)
    stride_pct: float = Field(
        default=78.0,
        ge=60,
        le=105,
        description="Back ankle at back-foot contact to front ankle at front-foot contact",
    )
    lateral_flexion_deg: float = Field(default=15.0, ge=0, le=50)
    release_height_pct: float = Field(
        default=118.0, ge=95, le=135, description="Estimated ball height (wrist + hand)"
    )
    runup_decel_pct: float = Field(default=5.0, ge=0, le=50)
    follow_through_pct: float = Field(default=80.0, ge=10, le=120)
    front_arm_ffc_deg: float = Field(
        default=70.0, ge=-90, le=120, description="Front upper-arm elevation above horizontal"
    )
    noise_px: float = Field(default=0.0, ge=0)
    dropout_rate: float = Field(default=0.0, ge=0, lt=1)
    seed: int = 0
    include_z: bool = True


class SynthTruth(BaseModel):
    """Values measured on the clean (noise-free) generated motion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    elbow_extension_deg: float
    counter_rotation_deg: float
    knee_flex_change_deg: float
    stride_pct: float
    lateral_flexion_deg: float
    release_height_pct: float
    runup_decel_pct: float
    follow_through_pct: float
    hip_shoulder_sep_bfc_deg: float
    height_px: float


class SynthResult(BaseModel):
    """A generated sequence with its ground truth."""

    model_config = ConfigDict(extra="forbid")

    params: SynthParams
    sequence: PoseSequence
    phases: PhaseEvents
    truth: SynthTruth


# --- small helpers --------------------------------------------------------------------------


def _smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def _unit(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def _yaw_dir(yaw_deg: float) -> FloatArray:
    """Right-to-left body line for a given yaw (0 = side-on along +X, 90 = front-on)."""
    r = np.radians(yaw_deg)
    return np.array([np.cos(r), 0.0, np.sin(r)])


class _Curve:
    """Monotone cubic (PCHIP) curve through keyframes, held constant outside them."""

    def __init__(self, points: Iterable[tuple[float, float]]) -> None:
        ts, vs = zip(*sorted(points), strict=True)
        self.t0, self.t1 = float(ts[0]), float(ts[-1])
        self._interp = PchipInterpolator(np.array(ts), np.array(vs))

    def __call__(self, t: float) -> float:
        return float(self._interp(min(max(t, self.t0), self.t1)))

    def many(self, t: FloatArray) -> FloatArray:
        """Evaluate at many times at once."""
        return np.asarray(self._interp(np.clip(t, self.t0, self.t1)), dtype=np.float64)


def _pchip(points: Iterable[tuple[float, float]]) -> _Curve:
    return _Curve(points)


def _planar(angle_deg: float) -> FloatArray:
    """Unit vector in the sagittal (X-Y) plane; 0 = forward, 90 = up."""
    r = np.radians(angle_deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def _two_link_ik(
    root: FloatArray, target: FloatArray, a: float, b: float, hint: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Place the middle joint of a 2-link chain; unreachable targets are pulled in.

    Returns ``(middle_joint, end_joint)``. Segment lengths are always exactly ``a`` and ``b``.
    """
    span = target - root
    dist = float(np.linalg.norm(span))
    e = span / dist if dist > 1e-12 else np.array([0.0, -1.0, 0.0])
    dist = min(max(dist, abs(a - b) + 1e-6), (a + b) * (1.0 - 1e-6))
    end = root + e * dist
    along = (a * a - b * b + dist * dist) / (2.0 * dist)
    perp_len = float(np.sqrt(max(a * a - along * along, 0.0)))
    perp = hint - float(np.dot(hint, e)) * e
    perp = _unit(perp) if np.linalg.norm(perp) > 1e-9 else np.array([0.0, 0.0, 1.0])
    return root + along * e + perp_len * perp, end


def _knee_span(knee_angle_deg: float) -> float:
    """Hip-to-ankle distance for an interior knee angle."""
    c = np.cos(np.radians(knee_angle_deg))
    return float(np.sqrt(THIGH**2 + SHANK**2 - 2.0 * THIGH * SHANK * c))


def _angle_at(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """Interior angle ABC in degrees."""
    u, v = a - b, c - b
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# --- the action -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plant:
    t_land: float
    t_lift: float
    pos: tuple[float, float]  # (x, z) of the ankle on the ground
    lift_h: float = 0.10  # swing height of the swing that ends with this plant


@dataclass(frozen=True)
class _Timeline:
    steps: tuple[float, ...]
    takeoff: float
    bound: float
    bfc: float
    ffc: float
    arm_horizontal: float
    release: float
    end: float

    @property
    def decel_start(self) -> float:
        return self.steps[N_RUNUP_STEPS - 6]


def _timeline() -> _Timeline:
    steps = tuple(T_STAND + 0.25 + k * STEP_PERIOD for k in range(N_RUNUP_STEPS))
    takeoff = steps[-1] + CONTACT
    bound = takeoff + BOUND_RISE
    bfc = bound + BOUND_FALL
    ffc = bfc + BFC_TO_FFC
    release = ffc + FFC_TO_RELEASE
    return _Timeline(
        steps=steps,
        takeoff=takeoff,
        bound=bound,
        bfc=bfc,
        ffc=ffc,
        arm_horizontal=release - AH_BEFORE_RELEASE,
        release=release,
        end=release + AFTER_RELEASE,
    )


class _Action:
    """Continuous-time right-arm bowling action in world coordinates (H = 1)."""

    def __init__(self, p: SynthParams) -> None:
        self.p = p
        tl = self.tl = _timeline()
        stride = p.stride_pct / 100.0
        hip_b, sh_b = ACTION_YAWS[p.action_type]
        self.hip_yaw_bfc, self.sh_yaw_bfc = hip_b, sh_b
        cr_min_t = tl.bfc + 0.6 * BFC_TO_FFC

        self.sh_yaw = _pchip(
            [
                (0.0, 88.0),
                (tl.takeoff, 88.0),
                (tl.bfc, sh_b),
                (cr_min_t, sh_b - p.counter_rotation_deg),
                (tl.release, 100.0),
                (tl.release + 0.3, 150.0),
                (tl.end, 150.0),
            ]
        )
        self.hip_yaw = _pchip(
            [
                (0.0, 88.0),
                (tl.takeoff, 88.0),
                (tl.bfc, hip_b),
                (tl.ffc, hip_b + 15.0),
                (tl.release, max(hip_b + 20.0, 85.0)),
                (tl.release + 0.3, 130.0),
                (tl.end, 130.0),
            ]
        )
        self.lat_flex = _pchip(
            [
                (0.0, 0.0),
                (tl.ffc, 2.0),
                (tl.release, p.lateral_flexion_deg),
                (tl.release + 0.3, 0.5 * p.lateral_flexion_deg),
                (tl.end, 5.0),
            ]
        )
        self.fwd_flex = _pchip(
            [
                (0.0, 5.0),
                (T_STAND + 0.3, 12.0),
                (tl.takeoff, 10.0),
                (tl.bfc, 5.0),
                (tl.release, 20.0),
                (tl.release + 0.4, 45.0),
                (tl.end, 30.0),
            ]
        )
        self.arm_base = _pchip(
            [
                (0.0, 270.0),
                (tl.takeoff - 0.1, 268.0),
                (tl.bound, 250.0),
                (tl.bfc, 225.0),
                (tl.ffc, 200.0),
                (tl.arm_horizontal, 180.0),
                (tl.release, 90.0),
                (tl.release + 0.05, 15.0),
                (tl.release + 0.15, -45.0),
                (tl.release + 0.4, -95.0),
                (tl.end, -100.0),
            ]
        )
        self.front_arm_base = _pchip(
            [
                (0.0, -90.0),
                (tl.takeoff - 0.1, -90.0),
                (tl.bound, 40.0),
                (tl.bfc, 80.0),
                (tl.ffc, p.front_arm_ffc_deg),
                # Pull-down at well under half the bowling arm's angular speed, as in real
                # fast bowling; the front arm keeps folding in through follow-through.
                (tl.release, p.front_arm_ffc_deg - 45.0),
                (tl.release + 0.2, -60.0),
                (tl.release + 0.45, -95.0),
                (tl.end, -95.0),
            ]
        )

        # Pelvis forward speed; follow-through scaled to hit the requested hip travel.
        self._v_pre = [
            (0.0, 0.0),
            (T_STAND, 0.0),
            (T_STAND + 0.6, V_PEAK),
            (tl.decel_start, V_PEAK),
            (tl.bfc, V_PEAK * (1 - p.runup_decel_pct / 100.0)),
            (tl.ffc, 0.75 * V_PEAK * (1 - p.runup_decel_pct / 100.0)),
            (tl.release, 0.25 * V_PEAK),
        ]
        target_travel = p.follow_through_pct / 100.0 * stride
        self.follow_scale = self._solve_follow_scale(target_travel)
        self.speed, self.pelvis_x = self._speed_and_x(self.follow_scale)

        # Front foot placement chosen so the estimated ball height at release hits the target.
        self.front_knee = _pchip(
            [
                (tl.ffc, FRONT_KNEE_AT_FFC),
                (tl.release, FRONT_KNEE_AT_FFC - p.knee_flex_change_deg),
            ]
        )
        dx_rel = self._solve_front_reach(p.release_height_pct / 100.0)
        front_hip_rel = self._hip_joint(tl.release, "l", 0.0)
        self.front_foot = (float(front_hip_rel[0] + dx_rel), float(front_hip_rel[2]))
        self.back_foot = (self.front_foot[0] - stride, -0.03)
        self._check_reach(tl.ffc)

        y_ffc = self._pelvis_y_planted(tl.ffc)
        y_rel = self._pelvis_y_planted(tl.release)
        self._pelvis_y_base = _pchip(
            [
                (0.0, 0.52),
                (T_STAND, 0.52),
                (tl.takeoff, 0.50),
                (tl.bound, 0.60),
                (tl.bfc, 0.49),
                (tl.ffc, y_ffc),
                (tl.release, y_rel),
                (tl.release + 0.25, y_rel - 0.03),
                (tl.release + 0.6, 0.50),
                (tl.end, 0.51),
            ]
        )
        self.plants = self._plan_feet()

    # pelvis ---------------------------------------------------------------------------------

    def _speed_and_x(self, scale: float) -> tuple[_Curve, Callable[[float], float]]:
        tl = self.tl
        post = [
            (tl.release + 0.2, scale * 0.30 * V_PEAK),
            (tl.release + 0.5, scale * 0.10 * V_PEAK),
            (tl.release + 0.75, 0.0),
            (tl.end, 0.0),
        ]
        speed = _pchip(self._v_pre + post)
        grid = np.linspace(0.0, tl.end, int(tl.end * 4000) + 1)
        x = cumulative_trapezoid(speed.many(grid), grid, initial=0)

        def pelvis_x(t: float) -> float:
            return float(np.interp(t, grid, x))

        return speed, pelvis_x

    def _solve_follow_scale(self, target: float) -> float:
        tl = self.tl
        lo, hi = 0.0, 20.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, px = self._speed_and_x(mid)
            travel = px(tl.end) - px(tl.release)
            if travel < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _hip_joint(self, t: float, side: str, pelvis_y: float) -> FloatArray:
        u = _yaw_dir(self.hip_yaw(t))
        sign = 1.0 if side == "l" else -1.0
        centre = np.array([self.pelvis_x(t), pelvis_y, 0.0])
        return centre + sign * 0.5 * HIP_WIDTH * u

    def _front_horizontal(self, t: float) -> float:
        hip = self._hip_joint(t, "l", 0.0)
        return float(np.hypot(self.front_foot[0] - hip[0], self.front_foot[1] - hip[2]))

    def _pelvis_y_planted(self, t: float) -> float:
        """Pelvis height that gives the scripted front-knee angle with the front foot planted."""
        span = _knee_span(self.front_knee(t))
        horiz = self._front_horizontal(t)
        if horiz >= span:
            raise ValueError(
                f"Front leg cannot reach its foot at t={t:.3f}s (needs {horiz:.3f}H, leg "
                f"spans {span:.3f}H). Reduce stride_pct or knee_flex_change_deg."
            )
        return ANKLE_H + float(np.sqrt(span**2 - horiz**2))

    def _check_reach(self, t: float) -> None:
        self._pelvis_y_planted(t)

    def _solve_front_reach(self, ball_height: float) -> float:
        """Horizontal hip-to-front-ankle distance at release giving the target ball height."""
        tl = self.tl
        upper = self.upper_body(tl.release, np.array([self.pelvis_x(tl.release), 0.0, 0.0]))
        offset = float(upper["r_wrist"][1]) + HAND  # ball height above the pelvis
        span = _knee_span(self.front_knee(tl.release))
        needed = ball_height - offset - ANKLE_H
        lo_dx, hi_dx = 0.05, 0.40
        best = float(np.sqrt(max(span**2 - min(needed, span) ** 2, 0.0)))
        reach_hi = offset + ANKLE_H + float(np.sqrt(span**2 - lo_dx**2))
        reach_lo = offset + ANKLE_H + float(np.sqrt(span**2 - hi_dx**2))
        if not (reach_lo <= ball_height <= reach_hi):
            raise ValueError(
                f"release_height_pct={ball_height * 100:.1f} is not reachable with these "
                f"parameters; achievable range is {reach_lo * 100:.1f}-{reach_hi * 100:.1f}."
            )
        return best

    def pelvis(self, t: float) -> FloatArray:
        tl = self.tl
        if tl.ffc <= t <= tl.release:
            y = self._pelvis_y_planted(t)
        else:
            y = self._pelvis_y_base(t)
            if T_STAND <= t <= tl.takeoff:
                fade = _smoothstep((t - T_STAND) / 0.3) * _smoothstep((tl.takeoff - t) / 0.15)
                y += 0.012 * fade * np.cos(2 * np.pi * (t - tl.steps[0]) / STEP_PERIOD)
        return np.array([self.pelvis_x(t), y, 0.0])

    # feet -----------------------------------------------------------------------------------

    def _plan_feet(self) -> dict[str, list[_Plant]]:
        tl = self.tl
        px = self.pelvis_x
        plants: dict[str, list[_Plant]] = {"l": [], "r": []}
        for k, t in enumerate(tl.steps):
            foot = "l" if (N_RUNUP_STEPS - 1 - k) % 2 == 0 else "r"
            z = 0.05 if foot == "l" else -0.05
            plants[foot].append(_Plant(t, t + CONTACT, (px(t) + 0.10, z)))
        first = {f: plants[f][0].t_land for f in plants}
        for foot, z in (("l", 0.06), ("r", -0.06)):
            lift = max(T_STAND, first[foot] - 0.8 * STEP_PERIOD)
            plants[foot].insert(0, _Plant(-1.0, lift, (px(0.0), z)))
        plants["r"].append(_Plant(tl.bfc, tl.ffc + 0.03, self.back_foot, 0.15))
        plants["l"].append(_Plant(tl.ffc, tl.release + 0.25, self.front_foot, 0.15))
        t_r = tl.release + 0.4
        plants["r"].append(_Plant(t_r, tl.end + 1.0, (px(t_r) + 0.12, -0.05), 0.12))
        t_l = tl.release + 0.65
        plants["l"].append(_Plant(t_l, tl.end + 1.0, (px(t_l) + 0.10, 0.05), 0.08))
        return plants

    def ankle_target(self, foot: str, t: float) -> FloatArray:
        seq = self.plants[foot]
        for i, pl in enumerate(seq):
            if pl.t_land <= t <= pl.t_lift:
                return np.array([pl.pos[0], ANKLE_H, pl.pos[1]])
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if nxt is not None and pl.t_lift < t < nxt.t_land:
                s = _smoothstep((t - pl.t_lift) / (nxt.t_land - pl.t_lift))
                x = pl.pos[0] + s * (nxt.pos[0] - pl.pos[0])
                z = pl.pos[1] + s * (nxt.pos[1] - pl.pos[1])
                y = ANKLE_H + nxt.lift_h * float(np.sin(np.pi * s))
                return np.array([x, y, z])
        last = seq[-1]
        return np.array([last.pos[0], ANKLE_H, last.pos[1]])

    # upper body -----------------------------------------------------------------------------

    def _swing(self, t: float) -> float:
        tl = self.tl
        fade = _smoothstep((t - T_STAND) / 0.3) * _smoothstep((tl.takeoff - 0.1 - t) / 0.2)
        return 25.0 * fade * float(np.sin(np.pi * (t - tl.steps[0]) / STEP_PERIOD))

    def elbow_angle(self, t: float) -> float:
        tl = self.tl
        flexed = BASE_ELBOW - self.p.elbow_extension_deg
        if t <= tl.arm_horizontal:
            return flexed
        if t >= tl.release:
            return BASE_ELBOW
        s = _smoothstep((t - tl.arm_horizontal) / (tl.release - tl.arm_horizontal))
        return flexed + s * self.p.elbow_extension_deg

    def upper_body(self, t: float, pelvis: FloatArray) -> dict[str, FloatArray]:
        """Shoulders, arms and head for a pelvis centre position."""
        up_world = np.array([0.0, 1.0, 0.0])
        u_flat = _yaw_dir(self.sh_yaw(t))
        f_flat = np.cross(up_world, u_flat)
        trunk_up = _unit(
            up_world
            + np.tan(np.radians(self.lat_flex(t))) * u_flat
            + np.tan(np.radians(self.fwd_flex(t))) * f_flat
        )
        u = _unit(u_flat - float(np.dot(u_flat, trunk_up)) * trunk_up)
        f = np.cross(trunk_up, u)
        sh_c = pelvis + TRUNK * trunk_up
        pts: dict[str, FloatArray] = {
            "l_shoulder": sh_c + 0.5 * SHOULDER_WIDTH * u,
            "r_shoulder": sh_c - 0.5 * SHOULDER_WIDTH * u,
        }
        head = sh_c + (NECK + HEAD_CENTRE) * trunk_up
        pts["nose"] = head + 0.06 * f - 0.01 * trunk_up
        pts["l_eye"] = head + 0.045 * f + 0.015 * trunk_up + 0.03 * u
        pts["r_eye"] = head + 0.045 * f + 0.015 * trunk_up - 0.03 * u
        pts["l_ear"] = head + 0.07 * u
        pts["r_ear"] = head - 0.07 * u

        beta = self.arm_base(t) + self._swing(t)
        elbow = pts["r_shoulder"] + UPPER_ARM * _planar(beta)
        pts["r_elbow"] = elbow
        pts["r_wrist"] = elbow + FOREARM * _planar(beta + 180.0 - self.elbow_angle(t))

        beta_f = self.front_arm_base(t) - self._swing(t)
        f_elbow = pts["l_shoulder"] + UPPER_ARM * _planar(beta_f)
        pts["l_elbow"] = f_elbow
        pts["l_wrist"] = f_elbow + FOREARM * _planar(beta_f + 180.0 - FRONT_ELBOW)
        return pts

    def pose(self, t: float) -> dict[str, FloatArray]:
        """All 21 keypoints at time ``t`` (right-arm bowler, world coordinates)."""
        pelvis = self.pelvis(t)
        pts = self.upper_body(t, pelvis)
        fwd = np.array([1.0, 0.0, 0.0])
        for side in ("l", "r"):
            hip = self._hip_joint(t, side, float(pelvis[1]))
            knee, ankle = _two_link_ik(hip, self.ankle_target(side, t), THIGH, SHANK, fwd)
            pts[f"{side}_hip"], pts[f"{side}_knee"], pts[f"{side}_ankle"] = hip, knee, ankle
            pts[f"{side}_heel"] = ankle + np.array([-HEEL_BACK, -HEEL_DOWN, 0.0])
            pts[f"{side}_toe"] = ankle + np.array([TOE_FWD, -TOE_DOWN, 0.0])
        return pts


# --- generation -----------------------------------------------------------------------------

_NAME_INDEX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}


def _mirror_names(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def world_motion(params: SynthParams) -> tuple[FloatArray, _Timeline, _Action]:
    """Clean world-space motion, shape ``(frames, 21, 3)`` in height units, mirrored for
    left-arm bowlers. Also returns the timeline and the action model."""
    action = _Action(params)
    tl = action.tl
    n_frames = int(np.floor(tl.end * params.fps)) + 1
    world = np.zeros((n_frames, len(KEYPOINT_NAMES), 3))
    for f in range(n_frames):
        pts = action.pose(f / params.fps)
        for name, xyz in pts.items():
            world[f, _NAME_INDEX[name]] = xyz
    if params.bowling_arm == "left":
        swapped = np.empty_like(world)
        for name, i in _NAME_INDEX.items():
            swapped[:, _NAME_INDEX[_mirror_names(name)]] = world[:, i]
        swapped[:, :, 2] *= -1.0
        world = swapped
    return world, tl, action


def _project(world: FloatArray, view: SynthView) -> FloatArray:
    """Orthographic camera: returns (img_x, img_up, depth) in height units."""
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    if view == "side":
        return np.stack([x, y, z], axis=-1)
    if view == "front":  # camera at the batter's end looking back down the pitch
        return np.stack([-z, y, -x], axis=-1)
    return np.stack([z, y, x], axis=-1)  # behind: camera behind the bowler


def _ground_truth(
    world: FloatArray, tl: _Timeline, params: SynthParams, action: _Action
) -> tuple[PhaseEvents, SynthTruth]:
    fps = params.fps
    n = world.shape[0]
    idx = _NAME_INDEX
    bow = "r" if params.bowling_arm == "right" else "l"
    front = "l" if params.bowling_arm == "right" else "r"
    back = bow

    def clamp(f: int) -> int:
        return int(min(max(f, 0), n - 1))

    pelvis = 0.5 * (world[:, idx["l_hip"]] + world[:, idx["r_hip"]])
    speed = np.abs(np.gradient(pelvis[:, 0])) * fps
    peak = float(speed.max())
    run_up_start = int(np.argmax(speed > 0.2 * peak))
    # Contacts: the first frame with the foot on the ground.
    bfc = clamp(int(np.ceil(tl.bfc * fps - 1e-9)))
    ffc = clamp(int(np.ceil(tl.ffc * fps - 1e-9)))
    bound = run_up_start + int(np.argmax(pelvis[run_up_start : bfc + 1, 1]))

    wrist = world[:, idx[f"{bow}_wrist"]]
    r0 = clamp(round(tl.release * fps))
    lo, hi = clamp(r0 - 3), clamp(r0 + 3)
    release = lo + int(np.argmax(wrist[lo : hi + 1, 1]))

    sh = world[:, idx[f"{bow}_shoulder"]]
    el = world[:, idx[f"{bow}_elbow"]]
    elev = np.degrees(np.arcsin(np.clip((el[:, 1] - sh[:, 1]) / UPPER_ARM, -1.0, 1.0)))
    crossings = [f for f in range(1, release) if elev[f - 1] < 0.0 <= elev[f]]
    if not crossings:
        raise RuntimeError("generator bug: the bowling arm never passes horizontal")
    arm_horizontal = crossings[-1]

    after = np.nonzero(speed[release + 1 :] < 0.15 * peak)[0]
    follow_end = release + 1 + int(after[0]) if after.size else n - 1

    phases = PhaseEvents(
        run_up_start=run_up_start,
        bound=bound,
        bfc=bfc,
        ffc=ffc,
        arm_horizontal=arm_horizontal,
        release=release,
        follow_through_end=follow_end,
        confidence=1.0,
    )
    if phases.ordering_violations():
        raise RuntimeError(f"generator bug: phase order {phases.ordering_violations()}")

    elbow = np.array(
        [_angle_at(sh[f], el[f], wrist[f]) for f in range(arm_horizontal, release + 1)]
    )
    k_max = int(np.argmax(elbow))
    extension = float(elbow[k_max] - elbow[: k_max + 1].min())

    def knee(f: int) -> float:
        return _angle_at(
            world[f, idx[f"{front}_hip"]],
            world[f, idx[f"{front}_knee"]],
            world[f, idx[f"{front}_ankle"]],
        )

    def line_yaw(f: int, joint: str) -> float:
        d = world[f, idx[f"{front}_{joint}"]] - world[f, idx[f"{back}_{joint}"]]
        z = d[2] if params.bowling_arm == "right" else -d[2]
        return float(np.degrees(np.arctan2(z, d[0])))

    sh_yaws = [line_yaw(f, "shoulder") for f in range(bfc, ffc + 1)]
    # Delivery stride: back ankle at back-foot contact to front ankle at front-foot contact.
    stride = abs(float(world[ffc, idx[f"{front}_ankle"], 0] - world[bfc, idx[f"{back}_ankle"], 0]))
    decel_from = clamp(round(tl.decel_start * fps))
    window = speed[decel_from : bfc + 1]
    runup_peak = float(speed[: bfc + 1].max())
    truth = SynthTruth(
        elbow_extension_deg=extension,
        counter_rotation_deg=sh_yaws[0] - min(sh_yaws),
        knee_flex_change_deg=knee(ffc) - knee(release),
        stride_pct=100.0 * stride,
        lateral_flexion_deg=float(action.lat_flex(release / fps)),
        release_height_pct=100.0 * (float(wrist[release, 1]) + HAND),
        runup_decel_pct=100.0 * (runup_peak - float(window.min())) / runup_peak,
        follow_through_pct=100.0 * float(pelvis[-1, 0] - pelvis[release, 0]) / stride,
        hip_shoulder_sep_bfc_deg=line_yaw(bfc, "hip") - line_yaw(bfc, "shoulder"),
        height_px=params.height_px,
    )
    return phases, truth


MARGIN_H = 0.25  # empty border around the figure, in heights

# Requested parameter -> (truth field, tolerance). Generation fails loudly if the body can't
# produce the requested value instead of silently returning something else.
REACH_CHECKS: dict[str, float] = {
    "elbow_extension_deg": 0.5,
    "counter_rotation_deg": 2.0,
    "knee_flex_change_deg": 0.5,
    "stride_pct": 1.0,
    "release_height_pct": 1.0,
    "runup_decel_pct": 1.0,
    "follow_through_pct": 2.0,
}


# Everything except stride is measured at specific frames, so below this frame rate the
# frame grid is too coarse to compare against the request; only stride is checked there.
SAMPLED_CHECKS: frozenset[str] = frozenset(set(REACH_CHECKS) - {"stride_pct"})
SAMPLED_CHECK_MIN_FPS = 100.0


def _check_truth(p: SynthParams, truth: SynthTruth) -> None:
    for name, tol in REACH_CHECKS.items():
        if name in SAMPLED_CHECKS and p.fps < SAMPLED_CHECK_MIN_FPS:
            continue
        wanted, got = getattr(p, name), getattr(truth, name)
        if abs(wanted - got) > tol:
            raise ValueError(
                f"{name}={wanted:g} could not be produced (got {got:.2f}); the combination "
                "of parameters is outside what the body model can reach."
            )


def generate(params: SynthParams | None = None) -> SynthResult:
    """Generate a synthetic delivery as a ``PoseSequence`` plus ground truth."""
    p = params or SynthParams()
    world, tl, action = world_motion(p)
    phases, truth = _ground_truth(world, tl, p, action)
    _check_truth(p, truth)

    img = _project(world, p.view)
    s = p.height_px
    x_min, x_max = float(img[..., 0].min()), float(img[..., 0].max())
    y_max = float(img[..., 1].max())
    width = int(np.ceil(((x_max - x_min) + 2 * MARGIN_H) * s / 2.0) * 2)
    height = int(np.ceil((y_max + 2 * MARGIN_H) * s / 2.0) * 2)
    px = (img[..., 0] - x_min + MARGIN_H) * s
    py = (y_max + MARGIN_H - img[..., 1]) * s
    pz = img[..., 2] * s

    rng = np.random.default_rng(p.seed)
    if p.noise_px > 0:
        px = px + rng.normal(0.0, p.noise_px, px.shape)
        py = py + rng.normal(0.0, p.noise_px, py.shape)
        conf = np.clip(rng.normal(0.9, 0.04, px.shape), 0.5, 1.0)
    else:
        conf = np.full(px.shape, 0.99)
    dropped = rng.random(px.shape) < p.dropout_rate

    frames: list[FramePose] = []
    for f in range(world.shape[0]):
        kps: dict[str, Keypoint | None] = {}
        for name, i in _NAME_INDEX.items():
            if dropped[f, i]:
                kps[name] = None
                continue
            kps[name] = Keypoint(
                x=float(px[f, i]),
                y=float(py[f, i]),
                z=float(pz[f, i]) if p.include_z else None,
                conf=float(conf[f, i]),
            )
        frames.append(FramePose(frame_idx=f, t=f / p.fps, keypoints=kps))

    seq = PoseSequence(
        fps=p.fps,
        width=width,
        height=height,
        view=p.view,
        bowling_arm=p.bowling_arm,
        frames=frames,
    )
    return SynthResult(params=p, sequence=seq, phases=phases, truth=truth)


# --- video ----------------------------------------------------------------------------------

BG_BGR = (30, 30, 30)
BONE_BGR = (230, 230, 230)
JOINT_BGR = (0, 200, 255)


def draw_frame(frame: FramePose, width: int, height: int, scale: float = 1.0) -> NDArray[np.uint8]:
    """Draw one stick-figure frame as a BGR image."""
    canvas = np.full((height, width, 3), BG_BGR, dtype=np.uint8)
    thickness = max(2, round(4 * scale))
    radius = max(2, round(5 * scale))

    def pt(name: str) -> tuple[int, int] | None:
        kp = frame.get(name)
        return None if kp is None else (round(kp.x), round(kp.y))

    for a, b in SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            cv2.line(canvas, pa, pb, BONE_BGR, thickness, cv2.LINE_AA)
    for name in KEYPOINT_NAMES:
        p = pt(name)
        if p is not None:
            cv2.circle(canvas, p, radius, JOINT_BGR, -1, cv2.LINE_AA)
    return canvas


def render_to_video(seq: PoseSequence, path: Path | str) -> Path:
    """Draw the sequence as a stick figure into an MP4 (mp4v) at the sequence fps.

    Frames are drawn one at a time; the writer is always released.

    Raises:
        OSError: if OpenCV cannot open a video writer for ``path``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(seq.fps), (seq.width, seq.height))
    try:
        if not writer.isOpened():
            raise OSError(f"OpenCV could not open a video writer for {out}")
        scale = seq.height / 600.0
        for frame in seq.frames:
            writer.write(draw_frame(frame, seq.width, seq.height, scale))
    finally:
        writer.release()
    return out
