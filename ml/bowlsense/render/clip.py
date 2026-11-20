"""Slow-motion release clips: the moments around release, cropped to the bowler, slowed down,
with the skeleton, the bowling-wrist path and live angles burned in. Exported as MP4 or GIF."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from bowlsense.analysis.legality import elbow_angles
from bowlsense.config import RenderSettings
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.body import joint
from bowlsense.io.video import Frame, iter_frames
from bowlsense.render.ffmpeg import H264Writer
from bowlsense.render.overlay import draw_pose, front_knee_angles
from bowlsense.schemas import DeliveryMetrics, PoseSequence

RELEASE_BGR = (74, 201, 255)  # amber
TRAIL_BGR = (58, 91, 255)  # signal red


def _crop_box(
    seq: PoseSequence, frames: range, width: int, height: int
) -> tuple[int, int, int, int]:
    """A box around every visible joint over ``frames`` with 25% padding, clamped to the
    image; falls back to the whole frame when no joints are visible."""
    by = {f.frame_idx: f for f in seq.frames}
    xs: list[float] = []
    ys: list[float] = []
    for i in frames:
        fp = by.get(i)
        if fp is None:
            continue
        for kp in fp.keypoints.values():
            if kp is not None:
                xs.append(kp.x)
                ys.append(kp.y)
    if not xs:
        return 0, 0, width, height
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.25 * max(x1 - x0, y1 - y0, 1.0)
    x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    x1, y1 = min(width, int(x1 + pad)), min(height, int(y1 + pad))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, width, height
    return x0, y0, x1, y1


def release_clip_frames(
    video: Path,
    seq: PoseSequence,
    delivery: DeliveryMetrics,
    settings: RenderSettings,
) -> tuple[list[Frame], float]:
    """Render the slow-motion release frames; returns (frames, output fps).

    ``seq`` is the cleaned pose for the whole video (frame_idx = video frame) and its ``fps``
    is the real-time rate, so slow-motion uploads are slowed relative to real time.
    """
    if seq.fps <= 0 or not seq.frames:
        raise InvalidInputError("The skeleton has no frames to make a clip from.")
    release = delivery.clip_start + delivery.phases.release
    first = max(0, release - round(settings.clip_before_s * seq.fps))
    last = release + round(settings.clip_after_s * seq.fps)
    # each source frame is repeated so the clip plays clip_slowdown x slower than real time
    repeat = max(1, math.ceil(settings.clip_slowdown * settings.clip_fps / seq.fps))
    hold = round(settings.clip_hold_s * settings.clip_fps)

    by = {f.frame_idx: i for i, f in enumerate(seq.frames)}
    elbow, knee = elbow_angles(seq), front_knee_angles(seq)
    elbow_name = joint(seq.bowling_arm, "bowling", "elbow")
    knee_name = joint(seq.bowling_arm, "front", "knee")
    wrist_name = joint(seq.bowling_arm, "bowling", "wrist")
    x0, y0, x1, y1 = _crop_box(seq, range(first, last + 1), seq.width, seq.height)
    scale = min(1.0, settings.clip_max_height / max(1, y1 - y0))
    out_w = max(2, int((x1 - x0) * scale) // 2 * 2)
    out_h = max(2, int((y1 - y0) * scale) // 2 * 2)

    trail: list[tuple[int, int]] = []
    out: list[Frame] = []
    for idx, frame in iter_frames(video, first, last + 1):
        i = by.get(idx)
        pose = seq.frames[i] if i is not None else None
        angles = (
            {elbow_name: ("Elbow", float(elbow[i])), knee_name: ("Knee", float(knee[i]))}
            if i is not None
            else {}
        )
        ms = (idx - release) / seq.fps * 1000.0
        label = "RELEASE" if idx == release else f"{ms:+.0f} ms from release"
        img = draw_pose(
            frame,
            pose,
            settings,
            label=f"{label}  |  {settings.clip_slowdown:.0f}x slow",
            angles=angles,
        )
        wrist = pose.get(wrist_name) if pose is not None else None
        if wrist is not None:
            trail.append((round(wrist.x), round(wrist.y)))
        s = max(1, round(min(img.shape[:2]) / 360))
        for a, b in pairwise(trail):
            cv2.line(img, a, b, TRAIL_BGR, s + 1, cv2.LINE_AA)
        if idx == release and wrist is not None:
            cv2.circle(img, trail[-1], 14 * s, RELEASE_BGR, 2 * s, cv2.LINE_AA)
        crop = img[y0:y1, x0:x1]
        crop = np.asarray(
            cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
        out.extend([crop] * (repeat + (hold if idx == release else 0)))
    if not out:
        raise InvalidInputError("The video has no frames around the release.")
    return out, settings.clip_fps


def write_release_clip(
    video: Path,
    seq: PoseSequence,
    delivery: DeliveryMetrics,
    settings: RenderSettings,
    out: Path,
    fmt: Literal["mp4", "gif"],
) -> Path:
    """Write the slow-motion release clip as an H.264 MP4 or an animated GIF."""
    frames, fps = release_clip_frames(video, seq, delivery, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        h, w = frames[0].shape[:2]
        with H264Writer(out, w, h, fps, settings.ffmpeg_preset, settings.crf) as writer:
            for f in frames:
                writer.write(f)
        return out
    h, w = frames[0].shape[:2]
    gw = min(settings.gif_width, w)
    gh = max(2, round(h * gw / w))
    step = max(1, round(fps / 15))  # GIFs at ~15 fps keep the file small
    images = [
        Image.fromarray(
            cv2.cvtColor(cv2.resize(f, (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        )
        for f in frames[::step]
    ]
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=round(1000 * step / fps), loop=0,
        optimize=True,
    )  # fmt: skip
    return out
