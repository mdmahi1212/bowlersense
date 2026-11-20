"""Body scale, bowling-arm detection, left/right role resolution and mirroring.

All distances that leave this package are expressed as % of standing height, so results
don't depend on how far the camera is from the bowler.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.config import BodySettings
from bowlsense.errors import InvalidInputError, PoseQualityError
from bowlsense.geometry.angles import FloatArray, keypoint_track, linear_speed
from bowlsense.schemas import Arm, FramePose, Keypoint, PoseSequence

Role = Literal["bowling", "non_bowling", "front", "back"]
_JOINTS = frozenset(
    {"shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe", "eye", "ear"}
)


class HeightEstimate(BaseModel):
    """Standing height in pixels and how it was measured."""

    model_config = ConfigDict(frozen=True)

    height_px: float
    frames_used: list[int]
    window: tuple[int, int]
    method: Literal["run-up", "segments"] = "run-up"


class BowlingArmResult(BaseModel):
    """Which arm bowls, from peak wrist speed."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    left_peak_px_s: float
    right_peak_px_s: float
    method: Literal["wrist_speed", "elbow_at_release", "profile", "only_visible_wrist"] = (
        "wrist_speed"
    )
    elbow_margin_px: float | None = None

    @property
    def ratio(self) -> float:
        """Faster wrist peak / slower wrist peak (>= 1); higher means a clearer decision."""
        lo = min(self.left_peak_px_s, self.right_peak_px_s)
        hi = max(self.left_peak_px_s, self.right_peak_px_s)
        return hi / lo if lo > 0 else float("inf")


def side_of(arm: Arm, role: Role) -> Literal["l", "r"]:
    """Body side prefix for a role. Right-arm bowler: bowling = back = right, front = left."""
    bowling: Literal["l", "r"] = "r" if arm == "right" else "l"
    other: Literal["l", "r"] = "l" if bowling == "r" else "r"
    return bowling if role in ("bowling", "back") else other


def joint(arm: Arm, role: Role, part: str) -> str:
    """Keypoint name for a role-relative joint, e.g. ``joint("left", "front", "ankle")`` ->
    ``"r_ankle"``."""
    if part not in _JOINTS:
        raise InvalidInputError(f"'{part}' is not a left/right joint.")
    return f"{side_of(arm, role)}_{part}"


def to_pct(distance_px: object, height_px: float) -> FloatArray:
    """Convert pixel distances to % of standing height."""
    if not np.isfinite(height_px) or height_px <= 0:
        raise InvalidInputError(f"height_px must be positive, got {height_px}.")
    return np.asarray(np.asarray(distance_px, dtype=np.float64) / height_px * 100.0)


def _peak_wrist_speeds(seq: PoseSequence) -> tuple[float, float, int]:
    """Peak speed of each wrist (``NaN`` for a wrist never seen) and the frame of the
    fastest wrist movement. Raises only when neither wrist is ever detected."""
    speeds = {
        side: linear_speed(keypoint_track(seq, f"{side}_wrist"), seq.fps) for side in ("l", "r")
    }
    both = np.fmax(speeds["l"], speeds["r"])
    if not np.isfinite(both).any():
        raise PoseQualityError(
            "Neither wrist was ever detected, so the bowling arm can't be determined. Film "
            "with the whole body in frame."
        )

    def peak(side: str) -> float:
        return float(np.nanmax(speeds[side])) if np.isfinite(speeds[side]).any() else float("nan")

    return peak("l"), peak("r"), int(np.nanargmax(both))


def detect_bowling_arm(seq: PoseSequence) -> BowlingArmResult:
    """The bowling arm is the side whose wrist reaches the higher peak speed.

    Run this on a cleaned sequence (see ``pose.smoothing``); single-frame keypoint glitches
    in raw detections can create fake speed peaks.

    When one wrist is never detected the visible one is taken (method
    ``only_visible_wrist``).

    Raises:
        PoseQualityError: if neither wrist is ever detected.
    """
    left, right, _ = _peak_wrist_speeds(seq)
    if not np.isfinite(left) or not np.isfinite(right):
        # one wrist is never seen: it's almost always the front arm hidden behind the body,
        # while the bowling arm swings overhead in view
        arm: Arm = "left" if np.isfinite(left) else "right"
        return BowlingArmResult(arm=arm, left_peak_px_s=left, right_peak_px_s=right,
                                method="only_visible_wrist")  # fmt: skip
    return BowlingArmResult(
        arm="right" if right >= left else "left", left_peak_px_s=left, right_peak_px_s=right
    )


def elbow_lift(seq: PoseSequence, arm: Arm, release: int, half_window: int = 2) -> float:
    """How much higher (pixels) the ``arm`` elbow is than the other elbow around ``release``.

    At release the bowling elbow is above the shoulder and the front elbow is down by the
    hip. Elbows blur far less than wrists at release, so this is robust at 30 fps.
    """
    sl = slice(max(0, release - half_window), release + half_window + 1)
    mine = keypoint_track(seq, joint(arm, "bowling", "elbow"))[sl, 1]
    other = keypoint_track(seq, joint(arm, "non_bowling", "elbow"))[sl, 1]
    diff = other - mine  # image y grows downward
    return float(np.nanmean(diff)) if np.isfinite(diff).any() else float("nan")


def detect_bowling_arm_at_release(
    seq: PoseSequence, release: int | None = None
) -> BowlingArmResult:
    """Bowling arm from elbow height at release (more robust on real footage than wrist
    speed, because elbows blur far less than wrists at release).

    Both elbows are compared at ONE release frame: ``release`` if given (e.g. from the
    learned phase model), otherwise the moment of fastest wrist movement. Falls back to peak
    wrist speed when the elbows aren't visible there.
    """
    speed = detect_bowling_arm(seq)
    frame = release if release is not None else _peak_wrist_speeds(seq)[2]
    margin = elbow_lift(seq, "right", frame)
    if not np.isfinite(margin):
        return speed
    arm: Arm = "right" if margin >= 0 else "left"
    return speed.model_copy(update={"arm": arm, "method": "elbow_at_release",
                                    "elbow_margin_px": abs(margin)})  # fmt: skip


def with_bowling_arm(seq: PoseSequence) -> PoseSequence:
    """Return ``seq`` with ``bowling_arm`` set from the detected arm."""
    return seq.model_copy(update={"bowling_arm": detect_bowling_arm(seq).arm})


def runup_window(
    seq: PoseSequence, settings: BodySettings, release: int | None = None
) -> tuple[int, int]:
    """``[start, stop)`` frames treated as run-up: from the start of the clip until
    ``runup_end_before_peak_s`` before ``release`` (default: the fastest wrist movement).

    For a video with several deliveries pass the *first* release, so the window doesn't
    include earlier follow-throughs.
    """
    peak = release if release is not None else _peak_wrist_speeds(seq)[2]
    stop = peak - round(settings.runup_end_before_peak_s * seq.fps)
    if stop < min_runup_frames(seq):  # short clip (reels, shorts): no run-up in view
        return 0, len(seq.frames)
    return 0, stop


def min_runup_frames(seq: PoseSequence) -> int:
    """Fewest frames before the delivery stride that count as a usable run-up."""
    return max(3, round(0.25 * seq.fps))


def has_runup(seq: PoseSequence, settings: BodySettings, release: int | None = None) -> bool:
    """True when the clip shows enough run-up before the delivery stride."""
    peak = release if release is not None else _peak_wrist_speeds(seq)[2]
    return peak - round(settings.runup_end_before_peak_s * seq.fps) >= min_runup_frames(seq)


SEGMENTS = (("ankle", "knee"), ("knee", "hip"), ("hip", "shoulder"))


def height_from_segments(seq: PoseSequence, settings: BodySettings) -> HeightEstimate:
    """Height from body-segment lengths (shank + thigh + trunk + shoulder-to-nose), which
    don't depend on posture, for clips without a run-up. Per frame the longer side of each
    segment is used (the nearer one is less foreshortened); the clip's upper percentile is
    taken, then scaled by ``segment_sum_frac`` (that chain is ~90% of standing height)."""
    totals = []
    for f in seq.frames:
        k = f.keypoints
        total = 0.0
        ok = True
        for a, b in SEGMENTS:
            lens = [
                np.hypot(k[f"{s}_{a}"].x - k[f"{s}_{b}"].x, k[f"{s}_{a}"].y - k[f"{s}_{b}"].y)  # type: ignore[union-attr]
                for s in ("l", "r")
                if k.get(f"{s}_{a}") is not None and k.get(f"{s}_{b}") is not None
            ]
            if not lens:
                ok = False
                break
            total += max(lens)
        nose = k.get("nose")
        sh = [k[f"{s}_shoulder"] for s in ("l", "r") if k.get(f"{s}_shoulder") is not None]
        if not ok or nose is None or not sh:
            continue
        total += min(np.hypot(nose.x - p.x, nose.y - p.y) for p in sh)  # type: ignore[union-attr]
        totals.append(total)
    if len(totals) < 3:
        raise PoseQualityError(
            "The bowler's legs, hips, shoulders and head are never visible together, so body "
            "height can't be measured. Keep the bowler's full body in frame."
        )
    height = float(np.percentile(totals, 90)) / settings.segment_sum_frac
    return HeightEstimate(height_px=height, frames_used=[], window=(0, len(seq.frames)),
                          method="segments")  # fmt: skip


def estimate_height_px(seq: PoseSequence, settings: BodySettings) -> HeightEstimate:
    """Standing height in pixels: vertical ankle-to-nose distance over the run-up (median of
    the top ``height_top_frac`` of frames) x ``height_factor``.

    The vertical distance is used because it is the same from side, front and behind views.
    Per frame the lower ankle is used (the stance leg is the extended one).

    Raises:
        PoseQualityError: if the nose or both ankles are missing in every frame (the run-up
            is preferred; the whole clip is used when the run-up has no full-body frame).
    """
    if not has_runup(seq, settings):
        return height_from_segments(seq, settings)
    start, stop = runup_window(seq, settings)

    def vertical(lo: int, hi: int) -> FloatArray:
        nose_y = keypoint_track(seq, "nose")[lo:hi, 1]
        d = [keypoint_track(seq, f"{side}_ankle")[lo:hi, 1] - nose_y for side in ("l", "r")]
        return np.asarray(np.fmax(d[0], d[1]), dtype=np.float64)

    per_frame = vertical(start, stop)
    valid = np.flatnonzero(np.isfinite(per_frame))
    if valid.size == 0:  # bowler not fully in view during the run-up: use the whole clip
        start, stop = 0, len(seq.frames)
        per_frame = vertical(start, stop)
        valid = np.flatnonzero(np.isfinite(per_frame))
    if valid.size == 0:  # never upright and fully in view: measure the body segments instead
        return height_from_segments(seq, settings)
    k = max(1, int(np.ceil(settings.height_top_frac * valid.size)))
    top = valid[np.argsort(per_frame[valid])[-k:]]
    height = float(np.median(per_frame[top])) * settings.height_factor
    if height <= 0:
        raise PoseQualityError("Measured body height is zero; keypoints look degenerate.")
    return HeightEstimate(
        height_px=height,
        frames_used=sorted(int(start + i) for i in top),
        window=(start, stop),
    )


def _swap_name(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def mirror_sequence(seq: PoseSequence) -> PoseSequence:
    """Mirror horizontally: ``x -> width - x``, left/right names swapped, bowling arm flipped.

    A left-arm bowler mirrored this way can be analysed with right-arm logic. Applying it
    twice returns the original sequence.
    """
    frames = []
    for frame in seq.frames:
        kps: dict[str, Keypoint | None] = {}
        for name, kp in frame.keypoints.items():
            kps[_swap_name(name)] = (
                None
                if kp is None
                else kp.model_copy(
                    update={"x": seq.width - kp.x, "z": None if kp.z is None else -kp.z}
                )
            )
        frames.append(FramePose(frame_idx=frame.frame_idx, t=frame.t, keypoints=kps))
    arm: Arm = "left" if seq.bowling_arm == "right" else "right"
    return seq.model_copy(update={"frames": frames, "bowling_arm": arm})
