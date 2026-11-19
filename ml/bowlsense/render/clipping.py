"""Split a long video into one clip per delivery around each detected release."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.config import ClippingSettings, RenderSettings
from bowlsense.errors import PhaseDetectionError
from bowlsense.geometry.angles import keypoint_track, linear_speed
from bowlsense.geometry.body import joint
from bowlsense.io.video import iter_frames, probe_video
from bowlsense.render.ffmpeg import H264Writer
from bowlsense.schemas import FramePose, PoseSequence


class ClipWindow(BaseModel):
    """Frames ``[start, stop)`` of the source video around one release."""

    model_config = ConfigDict(frozen=True)

    start: int
    stop: int
    release: int


def find_releases(seq: PoseSequence, settings: ClippingSettings) -> list[int]:
    """Release frames of every delivery in a cleaned sequence, in time order.

    A release is a peak of bowling-wrist speed while the wrist is above the shoulder, at
    least ``min_peak_frac`` of the fastest one, and at least ``min_gap_s`` from a faster
    peak. Each is refined to the wrist-height peak within +-3 frames.

    Raises:
        PhaseDetectionError: if no release is found.
    """
    arm = seq.bowling_arm
    wrist = keypoint_track(seq, joint(arm, "bowling", "wrist"))
    shoulder = keypoint_track(seq, joint(arm, "bowling", "shoulder"))
    speed = linear_speed(wrist, seq.fps)
    speed[~(wrist[:, 1] < shoulder[:, 1])] = np.nan
    if not np.isfinite(speed).any():
        raise PhaseDetectionError(
            "No delivery found: the bowling wrist is never above the shoulder."
        )
    floor = settings.min_peak_frac * float(np.nanmax(speed))
    gap = round(settings.min_gap_s * seq.fps)
    chosen: list[int] = []
    for f in np.argsort(np.nan_to_num(speed, nan=-1.0))[::-1]:
        if not np.isfinite(speed[f]) or speed[f] < floor:
            break
        if all(abs(int(f) - c) >= gap for c in chosen):
            chosen.append(int(f))
    n = len(seq.frames)
    refined = []
    for f in sorted(chosen):
        lo, hi = max(0, f - 3), min(n - 1, f + 3)
        heights = wrist[lo : hi + 1, 1]
        refined.append(lo + int(np.nanargmin(heights)) if np.isfinite(heights).any() else f)
    return refined


def clip_windows(
    releases: list[int], n_frames: int, fps: float, settings: ClippingSettings
) -> list[ClipWindow]:
    """``[release - before, release + after)`` per release, clamped to the video."""
    before = round(settings.before_release_s * fps)
    after = round(settings.after_release_s * fps)
    return [
        ClipWindow(start=max(0, r - before), stop=min(n_frames, r + after + 1), release=r)
        for r in releases
    ]


def sub_sequence(seq: PoseSequence, window: ClipWindow) -> PoseSequence:
    """The frames of one clip, renumbered from 0 (frame ``k`` = video frame
    ``window.start + k``)."""
    frames = [
        FramePose(frame_idx=i, t=i / seq.fps, keypoints=f.keypoints)
        for i, f in enumerate(seq.frames[window.start : window.stop])
    ]
    return seq.model_copy(update={"frames": frames})


def cut_clip(video: Path, window: ClipWindow, out: Path, settings: RenderSettings) -> Path:
    """Write the clip's frames as their own H.264 MP4 (frame-accurate)."""
    info = probe_video(video)
    with H264Writer(out, info.width, info.height, info.fps, settings.ffmpeg_preset,
                    settings.crf) as writer:  # fmt: skip
        for _, frame in iter_frames(video, window.start, window.stop):
            writer.write(frame)
    return out
