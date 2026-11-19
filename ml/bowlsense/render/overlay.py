"""Overlay video: skeleton, confidence-coloured joints, live elbow and front-knee angles, a
phase banner, and a red tint on frames inside a flagged legality window."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.analysis.legality import elbow_angles
from bowlsense.config import RenderSettings
from bowlsense.geometry.angles import FloatArray, joint_angle, keypoint_track
from bowlsense.geometry.body import joint
from bowlsense.io.video import Frame, iter_frames, probe_video
from bowlsense.render.ffmpeg import H264Writer
from bowlsense.schemas import (
    SKELETON_EDGES,
    FramePose,
    LegalityResult,
    PhaseEvents,
    PoseSequence,
)

BGR = tuple[int, int, int]
BONE: BGR = (235, 235, 235)
RED: BGR = (40, 40, 230)
AMBER: BGR = (0, 190, 255)
GREEN: BGR = (60, 200, 60)
BANNER_BG: BGR = (25, 25, 25)

EVENT_LABELS: dict[str, str] = {
    "run_up_start": "Run-up start",
    "bound": "Bound",
    "bfc": "Back-foot contact",
    "ffc": "Front-foot contact",
    "arm_horizontal": "Arm horizontal",
    "release": "Release",
    "follow_through_end": "Follow-through end",
}


def phase_label(frame: int, p: PhaseEvents) -> str:
    """Name of the phase a frame falls in (event names on the event frames themselves)."""
    for event, label in EVENT_LABELS.items():
        if frame == getattr(p, event):
            return label
    if p.run_up_start <= frame < p.bound:
        return "Run-up"
    if p.bound < frame < p.bfc:
        return "Bound"
    if p.bfc < frame < p.ffc:
        return "Delivery stride"
    if p.ffc < frame < p.release:
        return "Arm circle"
    if p.release < frame <= p.follow_through_end:
        return "Follow-through"
    return ""


def conf_colour(conf: float, settings: RenderSettings) -> BGR:
    """Red / amber / green by keypoint confidence."""
    if conf < settings.conf_low:
        return RED
    if conf < settings.conf_high:
        return AMBER
    return GREEN


def _scale(frame: Frame) -> float:
    return max(0.5, float(min(frame.shape[:2])) / 720.0)


def draw_pose(
    frame: Frame,
    pose: FramePose | None,
    settings: RenderSettings,
    *,
    label: str = "",
    angles: dict[str, tuple[str, float]] | None = None,
    tint: bool = False,
) -> Frame:
    """Return a copy of ``frame`` with the overlay drawn.

    ``angles`` maps a keypoint name to (short label, degrees) drawn next to that joint.
    """
    out = frame.copy()
    s = _scale(out)
    if tint:
        red = np.zeros_like(out)
        red[:, :] = RED
        out = np.asarray(
            cv2.addWeighted(out, 1.0 - settings.tint_alpha, red, settings.tint_alpha, 0.0),
            dtype=np.uint8,
        )
    if pose is not None:

        def pt(name: str) -> tuple[int, int] | None:
            kp = pose.get(name)
            return None if kp is None else (round(kp.x), round(kp.y))

        for a, b in SKELETON_EDGES:
            pa, pb = pt(a), pt(b)
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, BONE, max(1, round(2 * s)), cv2.LINE_AA)
        for kp in pose.keypoints.values():
            if kp is not None:
                cv2.circle(
                    out, (round(kp.x), round(kp.y)), max(2, round(4 * s)),
                    conf_colour(kp.conf, settings), -1, cv2.LINE_AA,
                )  # fmt: skip
        for name, (short, value) in (angles or {}).items():
            p = pt(name)
            if p is not None and np.isfinite(value):
                text = f"{short} {value:.0f}"
                org = (p[0] + round(8 * s), p[1] - round(8 * s))
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0),
                            max(2, round(4 * s)), cv2.LINE_AA)  # fmt: skip
                cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255),
                            max(1, round(1.5 * s)), cv2.LINE_AA)  # fmt: skip
    if label:
        h = round(34 * s)
        cv2.rectangle(out, (0, 0), (out.shape[1], h), BANNER_BG, -1)
        cv2.putText(out, label, (round(10 * s), round(24 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * s, (255, 255, 255), max(1, round(2 * s)), cv2.LINE_AA)  # fmt: skip
    return out


def front_knee_angles(seq: PoseSequence) -> FloatArray:
    """Front-knee angle per frame."""
    arm = seq.bowling_arm
    return joint_angle(
        *(keypoint_track(seq, joint(arm, "front", p)) for p in ("hip", "knee", "ankle"))
    )


class OverlaySegment(BaseModel):
    """One delivery inside the video: video frames ``[start, stop)``; its phases and
    legality are in clip numbering (clip frame 0 = video frame ``start``)."""

    model_config = ConfigDict(frozen=True)

    start: int
    stop: int
    phases: PhaseEvents
    legality: LegalityResult | None = None


class OverlayPainter:
    """Draws the per-frame overlay for a video and the deliveries found in it."""

    def __init__(
        self, seq: PoseSequence, segments: list[OverlaySegment], settings: RenderSettings
    ) -> None:
        """``seq`` is the cleaned pose for the whole video (frame_idx = video frame)."""
        self.seq, self.segments, self.settings = seq, segments, settings
        self.by_frame = {f.frame_idx: i for i, f in enumerate(seq.frames)}
        self.elbow = elbow_angles(seq)
        self.knee = front_knee_angles(seq)
        self.elbow_name = joint(seq.bowling_arm, "bowling", "elbow")
        self.knee_name = joint(seq.bowling_arm, "front", "knee")

    def segment_at(self, frame_idx: int) -> OverlaySegment | None:
        """The delivery a video frame belongs to, if any."""
        for seg in self.segments:
            if seg.start <= frame_idx < seg.stop:
                return seg
        return None

    def paint(self, frame_idx: int, frame: Frame) -> Frame:
        """The overlay for one video frame."""
        i = self.by_frame.get(frame_idx)
        pose = self.seq.frames[i] if i is not None else None
        angles = (
            {
                self.elbow_name: ("Elbow", float(self.elbow[i])),
                self.knee_name: ("Knee", float(self.knee[i])),
            }
            if i is not None
            else {}
        )
        label, tint = "", False
        seg = self.segment_at(frame_idx)
        if seg is not None:
            local = frame_idx - seg.start
            label = phase_label(local, seg.phases)
            if len(self.segments) > 1 and label:
                label = f"Ball {self.segments.index(seg) + 1}: {label}"
            if seg.legality is not None and seg.legality.flagged:
                tint = seg.phases.arm_horizontal <= local <= seg.phases.release
        return draw_pose(frame, pose, self.settings, label=label, angles=angles, tint=tint)


def render_overlay(
    video: Path,
    seq: PoseSequence,
    segments: list[OverlaySegment],
    out: Path,
    settings: RenderSettings,
) -> Path:
    """Write the overlay MP4 (H.264, faststart) for the whole video."""
    info = probe_video(video)
    painter = OverlayPainter(seq, segments, settings)
    with H264Writer(out, info.width, info.height, info.fps, settings.ffmpeg_preset,
                    settings.crf) as writer:  # fmt: skip
        for idx, frame in iter_frames(video):
            writer.write(painter.paint(idx, frame))
    return out
