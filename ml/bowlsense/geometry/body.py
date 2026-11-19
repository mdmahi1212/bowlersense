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


class BowlingArmResult(BaseModel):
    """Which arm bowls, from peak wrist speed."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    left_peak_px_s: float
    right_peak_px_s: float

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
    speeds = {}
    for side in ("l", "r"):
        s = linear_speed(keypoint_track(seq, f"{side}_wrist"), seq.fps)
        if not np.isfinite(s).any():
            raise PoseQualityError(
                f"The {'left' if side == 'l' else 'right'} wrist was never detected, so the "
                "bowling arm can't be determined. Film with the whole body in frame."
            )
        speeds[side] = s
    both = np.fmax(speeds["l"], speeds["r"])
    return float(np.nanmax(speeds["l"])), float(np.nanmax(speeds["r"])), int(np.nanargmax(both))


def detect_bowling_arm(seq: PoseSequence) -> BowlingArmResult:
    """The bowling arm is the side whose wrist reaches the higher peak speed.

    Run this on a cleaned sequence (see ``pose.smoothing``); single-frame keypoint glitches
    in raw detections can create fake speed peaks.

    Raises:
        PoseQualityError: if either wrist is never detected.
    """
    left, right, _ = _peak_wrist_speeds(seq)
    return BowlingArmResult(
        arm="right" if right >= left else "left", left_peak_px_s=left, right_peak_px_s=right
    )


def with_bowling_arm(seq: PoseSequence) -> PoseSequence:
    """Return ``seq`` with ``bowling_arm`` set from the detected arm."""
    return seq.model_copy(update={"bowling_arm": detect_bowling_arm(seq).arm})


def runup_window(seq: PoseSequence, settings: BodySettings) -> tuple[int, int]:
    """``[start, stop)`` frames treated as run-up: from the start of the clip until
    ``runup_end_before_peak_s`` before the fastest wrist movement (around release)."""
    _, _, peak = _peak_wrist_speeds(seq)
    stop = peak - round(settings.runup_end_before_peak_s * seq.fps)
    min_frames = max(3, round(0.25 * seq.fps))
    if stop < min_frames:
        raise PoseQualityError(
            "The clip starts too close to the delivery to measure the bowler's height. "
            "Start recording at least a second before the bowler reaches the crease."
        )
    return 0, stop


def estimate_height_px(seq: PoseSequence, settings: BodySettings) -> HeightEstimate:
    """Standing height in pixels: vertical ankle-to-nose distance over the run-up (median of
    the top ``height_top_frac`` of frames) x ``height_factor``.

    The vertical distance is used because it is the same from side, front and behind views.
    Per frame the lower ankle is used (the stance leg is the extended one).

    Raises:
        PoseQualityError: if the nose or both ankles are missing throughout the run-up.
    """
    start, stop = runup_window(seq, settings)
    nose_y = keypoint_track(seq, "nose")[start:stop, 1]
    dists = [keypoint_track(seq, f"{side}_ankle")[start:stop, 1] - nose_y for side in ("l", "r")]
    per_frame = np.fmax(dists[0], dists[1])
    valid = np.flatnonzero(np.isfinite(per_frame))
    if valid.size == 0:
        raise PoseQualityError(
            "The nose and ankles are never visible together during the run-up, so body "
            "height can't be measured. Keep the bowler's full body in frame."
        )
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
