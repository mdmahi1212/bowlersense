"""Slow-motion clips: find how much a video has been slowed down, and undo it for analysis.

Phone and broadcast "slo-mo" clips are often filmed at 120-480 fps and saved at 25-60 fps,
so the file's frame rate is wrong for real time: each frame is a few milliseconds, not
1/30 s. Every timing (phase windows, elbow screening, speeds) then needs the *real* frame
rate = file fps x slow-down factor.

The factor comes from the user, or is estimated from how fast the bowling arm turns about
the shoulder at its peak: about 1500-2000 deg/s for a pace bowler (``video.
slowmo_expected_arm_dps``). Arm rotation isn't affected by a camera panning with the bowler.
Where the arm circle can't be seen (front-on), the wrist's peak speed in body heights per
second is used instead (``video.slowmo_expected_wrist_hps``). If the measured peak is far
below the expected one, the clip is slowed down by about that ratio, snapped to a common
slow-motion rate.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.config import Settings
from bowlsense.errors import BowlSenseError
from bowlsense.geometry.angles import keypoint_track, linear_speed, valid_runs
from bowlsense.geometry.body import estimate_height_px, joint
from bowlsense.schemas import PoseSequence

log = logging.getLogger(__name__)


class SlowMotion(BaseModel):
    """How a clip's time base was set."""

    model_config = ConfigDict(frozen=True)

    factor: float  # real-time frame rate = file fps x factor
    source: str  # "user", "auto" or "none"
    wrist_peak_hps: float | None = None  # measured peak wrist speed, body heights / s (file fps)
    arm_peak_dps: float | None = None  # measured peak arm rotation, deg / s (file fps)


def wrist_peak_hps(cleaned: PoseSequence, settings: Settings) -> float | None:
    """Peak bowling-wrist speed in body heights per second at the file's frame rate."""
    try:
        height = estimate_height_px(cleaned, settings.body).height_px
    except BowlSenseError:
        return None
    wrist = keypoint_track(cleaned, joint(cleaned.bowling_arm, "bowling", "wrist"))
    speed = linear_speed(wrist, cleaned.fps)
    if not np.isfinite(speed).any() or height <= 0:
        return None
    return float(np.nanpercentile(speed, 99)) / height


def arm_peak_dps(cleaned: PoseSequence) -> float | None:
    """Peak rotation speed (deg/s at the file's frame rate) of the bowling arm (shoulder ->
    wrist line) in the image; ``None`` if the arm is never seen long enough."""
    arm = cleaned.bowling_arm
    shoulder = keypoint_track(cleaned, joint(arm, "bowling", "shoulder"))
    wrist = keypoint_track(cleaned, joint(arm, "bowling", "wrist"))
    ok = np.isfinite(shoulder).all(axis=1) & np.isfinite(wrist).all(axis=1)
    speeds = []
    for start, stop in valid_runs(ok):
        if stop - start < 3:
            continue
        d = wrist[start:stop] - shoulder[start:stop]
        ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        speeds.append(np.abs(np.gradient(ang)) * cleaned.fps)
    if not speeds:
        return None
    return float(np.degrees(np.percentile(np.concatenate(speeds), 99)))


def estimate_slowmo(cleaned: PoseSequence, settings: Settings) -> SlowMotion:
    """Auto-detect the slow-down factor from the arm's (or else the wrist's) peak speed."""
    v = settings.video
    arm = arm_peak_dps(cleaned) if cleaned.view in ("side", "unknown") else None
    wrist = wrist_peak_hps(cleaned, settings)
    if arm is not None and arm > 0:
        raw, how = v.slowmo_expected_arm_dps / arm, f"arm {arm:.0f} deg/s"
    elif wrist is not None and wrist > 0:
        raw, how = v.slowmo_expected_wrist_hps / wrist, f"wrist {wrist:.2f} heights/s"
    else:
        return SlowMotion(factor=1.0, source="none")
    if raw < v.slowmo_min_factor:
        return SlowMotion(factor=1.0, source="none", wrist_peak_hps=wrist, arm_peak_dps=arm)
    factor = min(v.slowmo_factors, key=lambda f: abs(np.log(f / raw)))
    log.info("slow motion: %s -> about %.1fx, using %gx", how, raw, factor)
    return SlowMotion(factor=float(factor), source="auto", wrist_peak_hps=wrist, arm_peak_dps=arm)


def retime(seq: PoseSequence, factor: float) -> PoseSequence:
    """``seq`` on the real-time base: fps x factor, timestamps / factor."""
    if factor == 1.0:
        return seq
    fps = seq.fps * factor
    frames = [f.model_copy(update={"t": f.frame_idx / fps}) for f in seq.frames]
    return seq.model_copy(update={"fps": fps, "frames": frames})
