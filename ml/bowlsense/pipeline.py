"""Orchestration: video -> pose sequence (front end) -> per-delivery analysis (modules A-E).

``run_pipeline`` is the full session pipeline: clipping, modules A-G, rendering, coach and PDF.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.analysis.alignment import assess_alignment
from bowlsense.analysis.compare import compare_to_references, consistency, load_references
from bowlsense.analysis.delivery import assess_ball
from bowlsense.analysis.fatigue import assess_fatigue, assess_workload, delivery_fatigue_input
from bowlsense.analysis.faults import assess_faults
from bowlsense.analysis.injury import assess_injury, injury_features
from bowlsense.analysis.legality import assess_legality
from bowlsense.config import Settings
from bowlsense.detect.ball import YoloBallDetector, track_ball
from bowlsense.detect.person import select_bowler, track_people_in_video
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.angles import FloatArray, keypoint_track
from bowlsense.geometry.body import (
    BowlingArmResult,
    detect_bowling_arm,
    detect_bowling_arm_at_release,
    estimate_height_px,
    joint,
    runup_window,
)
from bowlsense.geometry.pitch import PitchCalibration
from bowlsense.io.quality import check_pose, check_video, estimate_view
from bowlsense.io.video import VideoInfo, iter_frames, probe_video
from bowlsense.phases.heuristic import detect_phases
from bowlsense.phases.model import load_phase_model
from bowlsense.phases.seqclf import load_classifier
from bowlsense.pose.base import load_backend
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.render.clipping import ClipWindow, clip_windows, find_releases, sub_sequence
from bowlsense.schemas import (
    KEYPOINT_NAMES,
    AngleSeries,
    Arm,
    BallTrack,
    ConsistencyResult,
    DeliveryAnalysis,
    DeliveryMetrics,
    DeliveryTypeResult,
    FatigueResult,
    FramePose,
    PoseSequence,
    SessionResult,
    WorkloadResult,
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


def choose_bowling_arm(
    seq: PoseSequence, settings: Settings, known: Arm | None = None
) -> BowlingArmResult:
    """The bowling arm: the bowler profile's if known; else the elbow raised at release
    (release from the learned phase model, when enabled); else peak wrist speed.

    On real 30 fps match footage the elbow-at-learned-release rule was right 84.5% of the
    time (94% on the held-out test videos) against 59% for peak wrist speed.
    """
    if known is not None:
        return BowlingArmResult(arm=known, left_peak_px_s=float("nan"),
                                right_peak_px_s=float("nan"), method="profile")  # fmt: skip
    weights = settings.storage.models_path / settings.phases.model_file
    if settings.phases.use_model and weights.is_file():
        model = load_phase_model(weights)
        best = max(
            (model.predict(seq.model_copy(update={"bowling_arm": a})) for a in ("right", "left")),
            key=lambda p: p.release_confidence,
        )
        return detect_bowling_arm_at_release(seq, best.release)
    return detect_bowling_arm(seq)


def extract_pose(
    video: Path,
    settings: Settings,
    *,
    backend: str | None = None,
    allow_low_fps: bool = False,
    bowler_track_id: int | None = None,
    pose_json: Path | None = None,
    bowling_arm: Arm | None = None,
) -> FrontEndResult:
    """Video -> the bowler's pose sequence with view and bowling arm set.

    With ``pose_json`` the keypoints come from that file (a saved ``PoseSequence``) instead
    of the detector and pose model; the video is still checked. ``bowling_arm`` (from the
    bowler's profile) skips arm detection. With ``pose.lift_3d`` on, MotionBERT fills ``z``.
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
    arm_result = choose_bowling_arm(cleaned, settings, bowling_arm)
    arm = arm_result.arm
    cleaned = cleaned.model_copy(update={"bowling_arm": arm})
    first_release = find_releases(cleaned, settings.clipping)[0]
    window = runup_window(cleaned, settings.body, release=first_release)
    view = estimate_view(cleaned, window, settings.video)
    warnings += check_pose(seq, _visible_frames(seq), settings.video)
    log.info("front end: %s arm (%s), %s view, %d warning(s)", arm, arm_result.method, view,
             len(warnings))  # fmt: skip
    seq = seq.model_copy(update={"bowling_arm": arm, "view": view})
    if settings.pose.lift_3d:  # optional, feature-flagged; every module works in 2D without it
        from bowlsense.pose.lift3d import lift_sequence, load_lifter

        lifter = load_lifter(
            settings.storage.models_path / settings.pose.lift_model, settings.pose.lift_window
        )
        seq = lift_sequence(seq, lifter)
    return FrontEndResult(
        video=info,
        sequence=seq,
        warnings=warnings,
        bowler_track_id=track_id,
        backend=used_backend,
    )


def classify_delivery_type(
    seq: PoseSequence, release: int, weights: Path
) -> DeliveryTypeResult | None:
    """Pace or spin from the learned sequence classifier (2.5 s before to 0.5 s after
    release); ``None`` when no trained model is present."""
    if not weights.is_file():
        return None
    fps = seq.fps
    a, z = max(0, release - round(2.5 * fps)), min(len(seq.frames), release + round(0.5 * fps) + 1)
    probs = load_classifier(weights).predict_proba(
        seq.model_copy(update={"frames": seq.frames[a:z]})
    )
    p_fast = probs.get("fast", 0.0)
    return DeliveryTypeResult(delivery_type="fast" if p_fast >= 0.5 else "spin", p_fast=p_fast)


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
    models = settings.storage.models_path
    learned_release: int | None = None
    if settings.phases.use_model:
        learned_release = load_phase_model(models / settings.phases.model_file).predict(seq).release
    phases = detect_phases(seq, height, settings.phases, raw=unsmoothed, release=learned_release)
    type_file = models / settings.delivery.type_model_file
    delivery_type = classify_delivery_type(seq, phases.release, type_file)
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
        delivery_type=delivery_type,
        release_source="model" if learned_release is not None else "rules",
    )


class SessionAnalysis(BaseModel):
    """Every delivery found in one video, with its clip window, plus session fatigue."""

    windows: list[ClipWindow]
    deliveries: list[DeliveryAnalysis]
    fatigue: FatigueResult
    consistency: ConsistencyResult


def _with_ball(
    front: FrontEndResult,
    cleaned: PoseSequence,
    window: ClipWindow,
    delivery: DeliveryAnalysis,
    detector: YoloBallDetector,
    calibration: PitchCalibration | None,
    settings: Settings,
) -> DeliveryAnalysis:
    """Track the ball from this delivery's release and add module F's metrics."""
    release = window.start + delivery.phases.release
    wrist = keypoint_track(cleaned, joint(cleaned.bowling_arm, "bowling", "wrist"))[release]
    if not np.isfinite(wrist).all():
        track = BallTrack(points=[])
    else:
        frames = iter_frames(Path(front.video.path), max(0, release - 1))
        track = track_ball(
            frames, detector, release, (float(wrist[0]), float(wrist[1])), front.video.fps,
            settings.ball, settings.delivery.first_ball_within_frames,
        )  # fmt: skip
    ball = assess_ball(track, calibration, front.video.fps, settings.delivery)
    return delivery.model_copy(update={"ball": ball, "ball_track": track})


def analyze_session(
    front: FrontEndResult,
    settings: Settings,
    popping_crease: Sequence[Point] | None = None,
    calibration: PitchCalibration | None = None,
    track_ball_flight: bool = True,
) -> SessionAnalysis:
    """Split the video into deliveries around each release and analyse each one (modules
    A-E), then track the ball from each release (module F)."""
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
    if track_ball_flight:
        detector = YoloBallDetector(settings.storage.models_path, settings.detect, settings.ball)
        deliveries = [
            _with_ball(front, cleaned, w, d, detector, calibration, settings)
            for w, d in zip(windows, deliveries, strict=True)
        ]
    references = load_references(settings.compare.reference_path)
    if references:
        deliveries = [
            d.model_copy(
                update={
                    "comparisons": compare_to_references(
                        clean_sequence(sub_sequence(raw, w), settings.pose).model_copy(
                            update={"bowling_arm": raw.bowling_arm}
                        ),
                        d.phases,
                        references,
                        settings.compare,
                    )
                }
            )
            for w, d in zip(windows, deliveries, strict=True)
        ]
    fatigue = assess_fatigue([d.fatigue_input for d in deliveries], settings.fatigue)
    return SessionAnalysis(
        windows=windows,
        deliveries=deliveries,
        fatigue=fatigue,
        consistency=consistency(deliveries, settings.compare),
    )


# --- full session pipeline (Phase 12) -----------------------------------------------------------

ProgressFn = Callable[[float, str], None]


class PipelineOptions(BaseModel):
    """Per-run choices (CLI flags / API session fields)."""

    model_config = ConfigDict(frozen=True)

    backend: str | None = None
    allow_low_fps: bool = False
    bowler_track_id: int | None = None
    pose_json: Path | None = None
    bowling_arm: Arm | None = None
    calibration_points: list[list[float]] | None = None
    use_llm: bool = True
    recorded_on: date | None = None
    workload_history: list[tuple[date, int]] | None = None  # None: no bowler profile


def _series(name: str, values: FloatArray) -> AngleSeries:
    return AngleSeries(
        name=name,
        values=[float(v) if np.isfinite(v) else None for v in values],
        frames=list(range(len(values))),
    )


def delivery_angles(seq: PoseSequence) -> list[AngleSeries]:
    """The charted angle series of one (cleaned) delivery clip."""
    from bowlsense.analysis.legality import elbow_angles
    from bowlsense.render.overlay import front_knee_angles
    from bowlsense.render.report_pdf import trunk_lean

    return [
        _series("elbow", elbow_angles(seq)),
        _series("front_knee", front_knee_angles(seq)),
        _series("trunk_lean", trunk_lean(seq)),
    ]


def delivery_metrics(
    idx: int,
    analysis: DeliveryAnalysis,
    clip: PoseSequence,
    window: ClipWindow,
    keyframes: dict[str, str],
) -> DeliveryMetrics:
    """Flatten one ``DeliveryAnalysis`` into the ``DeliveryMetrics`` contract."""
    m = analysis.faults.measurements
    stride = m.get("OVERSTRIDE", m.get("UNDERSTRIDE"))
    ball = analysis.ball
    return DeliveryMetrics(
        delivery_idx=idx,
        phases=analysis.phases,
        action_type=analysis.alignment.action_type,
        legality=analysis.legality,
        faults=analysis.faults.faults,
        technique_score=analysis.faults.technique_score,
        injury=analysis.injury,
        speed_kmh=ball.speed_kmh if ball else None,
        length=ball.length if ball else None,
        line=ball.line if ball else None,
        pitch_xy=ball.pitch_xy if ball else None,
        release_height_pct=m.get("LOW_RELEASE"),
        stride_pct=stride,
        angles=delivery_angles(clip),
        clip_start=window.start,
        clip_stop=window.stop,
        alignment=analysis.alignment,
        fault_measurements=m,
        not_measured=analysis.faults.not_measured,
        delivery_type=analysis.delivery_type,
        release_source=analysis.release_source,
        ball_notes=ball.notes if ball else [],
        ball_track=analysis.ball_track,
        comparisons=analysis.comparisons,
        keyframes=keyframes,
    )


def similar_actions(deliveries: Sequence[DeliveryAnalysis]) -> list[tuple[str, float]]:
    """Reference actions by mean similarity over the session, best first."""
    sims: dict[str, list[float]] = {}
    for d in deliveries:
        for c in d.comparisons:
            sims.setdefault(c.name, []).append(c.similarity_pct)
    ranked = [(name, round(float(np.mean(v)), 1)) for name, v in sims.items()]
    return sorted(ranked, key=lambda kv: -kv[1])


def run_pipeline(
    video: Path,
    out_dir: Path,
    settings: Settings,
    *,
    session_id: str,
    options: PipelineOptions | None = None,
    progress: ProgressFn | None = None,
    track_ball_flight: bool = True,
) -> SessionResult:
    """Video -> every artifact (overlay, keyframes, charts, PDF, pose.json, result.json) in
    ``out_dir`` and the ``SessionResult`` (artifact paths relative to ``out_dir``)."""
    from bowlsense.coach.llm import generate_report
    from bowlsense.coach.prompts import summarise_session
    from bowlsense.geometry.pitch import calibrate
    from bowlsense.render.keyframes import export_keyframes
    from bowlsense.render.overlay import OverlayPainter, OverlaySegment, render_overlay
    from bowlsense.render.report_pdf import ReportDelivery, ReportInput, build_report

    opts = options or PipelineOptions()

    def report(pct: float, stage: str) -> None:
        log.info("pipeline %3.0f%% %s", pct, stage)
        if progress is not None:
            progress(pct, stage)

    report(2, "checking video and estimating pose")
    front = extract_pose(
        video, settings, backend=opts.backend, allow_low_fps=opts.allow_low_fps,
        bowler_track_id=opts.bowler_track_id, pose_json=opts.pose_json,
        bowling_arm=opts.bowling_arm,
    )  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pose.json").write_text(front.sequence.model_dump_json(), encoding="utf-8")
    report(45, "analysing deliveries")
    points = opts.calibration_points
    calibration = calibrate([(p[0], p[1]) for p in points]) if points else None
    session = analyze_session(
        front, settings, calibration=calibration, track_ball_flight=track_ball_flight
    )
    report(65, "rendering overlay video")
    cleaned = clean_sequence(front.sequence, settings.pose)
    segments = [
        OverlaySegment(start=w.start, stop=w.stop, phases=d.phases, legality=d.legality)
        for w, d in zip(session.windows, session.deliveries, strict=True)
    ]
    overlay = render_overlay(video, cleaned, segments, out_dir / "overlay.mp4", settings.render)
    report(80, "exporting keyframes")
    painter = OverlayPainter(cleaned, segments, settings.render)
    report_deliveries, metrics = [], []
    artifacts = {"overlay": overlay.name, "pose": "pose.json"}
    for i, (seg, delivery) in enumerate(zip(segments, session.deliveries, strict=True)):
        folder = out_dir / "keyframes" / f"delivery_{i + 1:02d}"
        keyframes = export_keyframes(video, painter, seg, folder, settings.render)
        rel = {ev: p.relative_to(out_dir).as_posix() for ev, p in keyframes.items()}
        clip = sub_sequence(cleaned, session.windows[i])
        report_deliveries.append(
            ReportDelivery(index=i, analysis=delivery, sequence=clip, keyframes=keyframes)
        )
        metrics.append(delivery_metrics(i, delivery, clip, session.windows[i], rel))
    report(88, "writing coach report")
    seq = front.sequence
    summary = summarise_session(
        session.deliveries, session.fatigue, session.consistency, seq.view, seq.bowling_arm,
        settings,
    )  # fmt: skip
    (out_dir / "coach_summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    artifacts["coach_summary"] = "coach_summary.json"
    coach = generate_report(summary, settings.coach, use_llm=opts.use_llm)
    report(94, "building PDF report")
    pdf = build_report(
        ReportInput(
            video_name=video.name, created=datetime.now(), view=seq.view,
            bowling_arm=seq.bowling_arm, fps=front.video.fps,
            quality_warnings=front.warnings, deliveries=report_deliveries,
            coach_text=coach.text,
        ),
        out_dir / "report.pdf", out_dir / "charts",
        settings.legality.elbow_extension_threshold_deg,
    )  # fmt: skip
    artifacts["report"] = pdf.name
    today = opts.recorded_on or datetime.now(UTC).date()
    n = len(session.deliveries)
    workload = (
        assess_workload(opts.workload_history, today, n, settings.workload)
        if opts.workload_history is not None
        else WorkloadResult(balls_this_session=n, note="No bowler profile: workload not tracked.")
    )
    result = SessionResult(
        session_id=session_id,
        deliveries=metrics,
        consistency_score=session.consistency.score,
        fatigue=session.fatigue,
        workload=workload,
        similar_pros=similar_actions(session.deliveries),
        coach_report=coach.text,
        quality_warnings=front.warnings,
        artifacts=artifacts,
        fps=front.video.fps,
        width=front.video.width,
        height=front.video.height,
        frame_count=front.video.frame_count,
        view=seq.view,
        bowling_arm=seq.bowling_arm,
        pose_backend=front.backend,
        bowler_track_id=front.bowler_track_id,
        consistency=session.consistency,
        coach_source=coach.source,
        coach_note=coach.note,
    )
    (out_dir / "result.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
    report(100, "done")
    return result
