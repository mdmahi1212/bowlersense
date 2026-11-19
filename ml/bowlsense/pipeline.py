"""Orchestration: video -> pose sequence (front end) -> per-delivery analysis (modules A-E).

Phase 12 extends this into the full session pipeline (clipping, rendering, coach, PDF).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bowlsense.analysis.alignment import assess_alignment
from bowlsense.analysis.fatigue import assess_fatigue, delivery_fatigue_input
from bowlsense.analysis.faults import assess_faults
from bowlsense.analysis.injury import assess_injury, injury_features
from bowlsense.analysis.legality import assess_legality
from bowlsense.config import Settings
from bowlsense.detect.person import select_bowler, track_people_in_video
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.body import detect_bowling_arm, estimate_height_px, runup_window
from bowlsense.io.quality import check_pose, check_video, estimate_view
from bowlsense.io.video import VideoInfo, iter_frames, probe_video
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.base import load_backend
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.render.clipping import ClipWindow, clip_windows, find_releases, sub_sequence
from bowlsense.schemas import (
    KEYPOINT_NAMES,
    DeliveryAnalysis,
    FatigueResult,
    FramePose,
    PoseSequence,
)

log = logging.getLogger(__name__)
Point = tuple[float, float]


class FrontEndResult(BaseModel):
    """The bowler's raw pose from a video, plus what was learned about the recording."""

    model_config = ConfigDict(frozen=True)

    video: VideoInfo
    sequence: PoseSequence  # raw keypoints; view and bowling arm filled in
    warnings: list[str]
    bowler_track_id: int | None
    backend: str


def _visible_frames(seq: PoseSequence) -> int:
    return sum(any(kp is not None for kp in f.keypoints.values()) for f in seq.frames)


def _pose_from_json(path: Path, info: VideoInfo) -> PoseSequence:
    if not path.is_file():
        raise InvalidInputError(f"Pose JSON not found: {path}")
    seq = PoseSequence.model_validate_json(path.read_text(encoding="utf-8"))
    if (seq.width, seq.height) != (info.width, info.height):
        raise InvalidInputError(
            f"Pose JSON is for a {seq.width}x{seq.height} video but {Path(info.path).name} is "
            f"{info.width}x{info.height}."
        )
    if abs(seq.fps - info.fps) > 0.5:
        raise InvalidInputError(
            f"Pose JSON is at {seq.fps:g} fps but the video is {info.fps:g} fps."
        )
    return seq


def extract_pose(
    video: Path,
    settings: Settings,
    *,
    backend: str | None = None,
    allow_low_fps: bool = False,
    bowler_track_id: int | None = None,
    pose_json: Path | None = None,
) -> FrontEndResult:
    """Video -> the bowler's pose sequence with view and bowling arm set.

    With ``pose_json`` the keypoints come from that file (a saved ``PoseSequence``) instead
    of the detector and pose model; the video is still checked.
    """
    info = probe_video(video)
    warnings = check_video(info, settings.video, allow_low_fps)
    track_id: int | None = None
    used_backend = "pose-json"
    if pose_json is not None:
        seq = _pose_from_json(pose_json, info)
    else:
        name = backend or settings.pose.backend
        models = settings.storage.models_path
        tracks = track_people_in_video(video, models, settings.detect)
        bowler = select_bowler(
            tracks, info.width, settings.detect.min_track_frames, bowler_track_id
        )
        track_id, used_backend = bowler.track_id, name
        pose = load_backend(name, settings.pose, models)
        empty = dict.fromkeys(KEYPOINT_NAMES)
        frames: list[FramePose] = []
        for idx, frame in iter_frames(video):
            box = bowler.box_at(idx)
            kps = pose.estimate(frame, box) if box is not None else dict(empty)
            frames.append(FramePose(frame_idx=idx, t=idx / info.fps, keypoints=kps))
        if not frames:
            raise InvalidInputError(f"No frames could be decoded from {video.name}.")
        seq = PoseSequence(fps=info.fps, width=info.width, height=info.height, frames=frames)

    cleaned = clean_sequence(seq, settings.pose)
    arm = detect_bowling_arm(cleaned).arm
    cleaned = cleaned.model_copy(update={"bowling_arm": arm})
    first_release = find_releases(cleaned, settings.clipping)[0]
    window = runup_window(cleaned, settings.body, release=first_release)
    view = estimate_view(cleaned, window, settings.video)
    warnings += check_pose(seq, _visible_frames(seq), settings.video)
    log.info("front end: %s arm, %s view, %d warning(s)", arm, view, len(warnings))
    return FrontEndResult(
        video=info,
        sequence=seq.model_copy(update={"bowling_arm": arm, "view": view}),
        warnings=warnings,
        bowler_track_id=track_id,
        backend=used_backend,
    )


def analyze_delivery(
    raw: PoseSequence,
    settings: Settings,
    *,
    delivery_idx: int = 0,
    popping_crease: Sequence[Point] | None = None,
    fatigue_index: float | None = None,
) -> DeliveryAnalysis:
    """Run phases and modules A-D on one delivery's raw pose sequence."""
    seq = clean_sequence(raw, settings.pose)
    unsmoothed = clean_sequence(raw, settings.pose, smooth=False)
    height = estimate_height_px(seq, settings.body).height_px
    phases = detect_phases(seq, height, settings.phases, raw=unsmoothed)
    legality = assess_legality(
        seq, phases, settings.legality, raw=unsmoothed, popping_crease=popping_crease
    )
    alignment = assess_alignment(seq, phases, settings.alignment, raw=unsmoothed)
    faults = assess_faults(seq, phases, height, settings, raw=unsmoothed)
    features = injury_features(seq, phases, alignment, faults, settings, fatigue_index)
    return DeliveryAnalysis(
        height_px=height,
        phases=phases,
        legality=legality,
        alignment=alignment,
        faults=faults,
        injury=assess_injury(features, settings.injury),
        fatigue_input=delivery_fatigue_input(delivery_idx, seq, phases, height, faults, settings),
    )


class SessionAnalysis(BaseModel):
    """Every delivery found in one video, with its clip window, plus session fatigue."""

    windows: list[ClipWindow]
    deliveries: list[DeliveryAnalysis]
    fatigue: FatigueResult


def analyze_session(
    front: FrontEndResult,
    settings: Settings,
    popping_crease: Sequence[Point] | None = None,
) -> SessionAnalysis:
    """Split the video into deliveries around each release and analyse each one."""
    raw = front.sequence
    cleaned = clean_sequence(raw, settings.pose)
    releases = find_releases(cleaned, settings.clipping)
    windows = clip_windows(releases, len(raw.frames), raw.fps, settings.clipping)
    log.info("session: %d deliver%s found", len(windows), "y" if len(windows) == 1 else "ies")
    deliveries = [
        analyze_delivery(
            sub_sequence(raw, w), settings, delivery_idx=i, popping_crease=popping_crease
        )
        for i, w in enumerate(windows)
    ]
    fatigue = assess_fatigue([d.fatigue_input for d in deliveries], settings.fatigue)
    return SessionAnalysis(windows=windows, deliveries=deliveries, fatigue=fatigue)
