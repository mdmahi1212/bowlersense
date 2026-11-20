"""Slow-motion release clips: the moments around release, cropped to the bowler, slowed down,
with the skeleton, the bowling-wrist path and live angles burned in. Exported as MP4 or GIF."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def _fit(frame: Frame, width: int, height: int) -> Frame:
    """Letterbox ``frame`` into width x height (black bars, aspect kept)."""
    h, w = frame.shape[:2]
    k = min(width / w, height / h)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    out = np.zeros((height, width, 3), np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y : y + nh, x : x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _title_card(width: int, height: int, title: str, subtitle: str) -> Frame:
    img = Image.new("RGB", (width, height), (11, 11, 10))
    draw = ImageDraw.Draw(img)
    big, small = _font(max(14, height // 12)), _font(max(10, height // 26))
    draw.text((width * 0.08, height * 0.40), title, font=big, fill=(242, 238, 229))
    draw.text((width * 0.08, height * 0.40 + height // 9), subtitle, font=small,
              fill=(255, 91, 58))  # fmt: skip
    draw.line([(width * 0.08, height * 0.36), (width * 0.3, height * 0.36)], fill=(179, 38, 30),
              width=max(2, height // 200))  # fmt: skip
    return np.asarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), dtype=np.uint8)


def write_highlight_reel(
    video: Path,
    seq: PoseSequence,
    picks: list[tuple[DeliveryMetrics, str, str]],
    settings: RenderSettings,
    out: Path,
) -> Path:
    """One MP4 of several deliveries' slow-motion release clips, each after a title card
    (``picks`` = (delivery, title, subtitle))."""
    if not picks:
        raise InvalidInputError("No deliveries to put in the reel.")
    clips = [(release_clip_frames(video, seq, d, settings), t, s) for d, t, s in picks]
    height = max(f[0][0].shape[0] for f, _, _ in clips)
    width = max(f[0][0].shape[1] for f, _, _ in clips)
    width, height = width // 2 * 2, height // 2 * 2
    fps = settings.clip_fps
    card = round(1.4 * fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    with H264Writer(out, width, height, fps, settings.ffmpeg_preset, settings.crf) as writer:
        for (frames, _), title, subtitle in clips:
            still = _title_card(width, height, title, subtitle)
            for _ in range(card):
                writer.write(still)
            for f in frames:
                writer.write(_fit(f, width, height))
    return out


def write_poster(
    video: Path,
    seq: PoseSequence,
    delivery: DeliveryMetrics,
    facts: list[tuple[str, str]],
    verdict: tuple[str, tuple[int, int, int]],
    title: str,
    settings: RenderSettings,
    out: Path,
) -> Path:
    """A shareable PNG: the release frame (skeleton and angles drawn, cropped to the bowler)
    beside a paper panel with the verdict and the key numbers."""
    release = delivery.clip_start + delivery.phases.release
    by = {f.frame_idx: i for i, f in enumerate(seq.frames)}
    i = by.get(release)
    frame = next((f for _, f in iter_frames(video, release, release + 1)), None)
    if frame is None:
        raise InvalidInputError("The release frame couldn't be read from the video.")
    pose = seq.frames[i] if i is not None else None
    elbow = elbow_angles(seq)
    knee = front_knee_angles(seq)
    angles = (
        {joint(seq.bowling_arm, "bowling", "elbow"): ("Elbow", float(elbow[i])),
         joint(seq.bowling_arm, "front", "knee"): ("Knee", float(knee[i]))}
        if i is not None else {}
    )  # fmt: skip
    img = draw_pose(frame, pose, settings, angles=angles)
    x0, y0, x1, y1 = _crop_box(seq, range(release, release + 1), seq.width, seq.height)
    photo = img[y0:y1, x0:x1]
    ph = 1200
    pw = max(2, int(photo.shape[1] * ph / photo.shape[0]))
    photo = np.asarray(cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.uint8)
    panel_w = 760
    canvas = Image.new("RGB", (pw + panel_w, ph), (242, 238, 229))
    canvas.paste(Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas)
    ink, muted, red = (22, 20, 15), (109, 102, 88), (179, 38, 30)
    x = pw + 60
    d.text((x, 60), "BowlSense", font=_font(40), fill=ink)
    d.line([(x, 120), (pw + panel_w - 60, 120)], fill=ink, width=2)
    d.text((x, 150), title, font=_font(30), fill=muted)
    d.text((x, 210), verdict[0], font=_font(96), fill=verdict[1])
    y = 360
    for label, value in facts:
        d.text((x, y), label.upper(), font=_font(20), fill=muted)
        d.text((x, y + 26), value, font=_font(52), fill=ink)
        y += 108
    d.line([(x, ph - 120), (pw + panel_w - 60, ph - 120)], fill=red, width=2)
    d.text((x, ph - 100), "2D video screening, not the ICC 3D test.", font=_font(20), fill=muted)
    d.text((x, ph - 70), "Injury load is not medical advice.", font=_font(20), fill=muted)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    return out
