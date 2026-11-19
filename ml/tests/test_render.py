"""Tests for rendering (Phase 8): H.264 writer, overlay, keyframes, clipping, PDF, CLI."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pypdf
import pytest

from bowlsense.cli import main as cli_main
from bowlsense.config import Settings, load_settings
from bowlsense.errors import PhaseDetectionError
from bowlsense.io.video import probe_video, read_frame
from bowlsense.pipeline import analyze_delivery
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.render.clipping import (
    ClipWindow,
    clip_windows,
    cut_clip,
    find_releases,
    sub_sequence,
)
from bowlsense.render.ffmpeg import (
    H264Writer,
    RenderError,
    even_size,
    mp4_top_level_boxes,
    probe_codec,
)
from bowlsense.render.keyframes import export_keyframes
from bowlsense.render.overlay import (
    AMBER,
    GREEN,
    RED,
    OverlayPainter,
    OverlaySegment,
    conf_colour,
    draw_pose,
    phase_label,
    render_overlay,
)
from bowlsense.render.report_pdf import (
    INJURY_NOTE,
    LEGALITY_NOTE,
    ReportDelivery,
    ReportInput,
    build_report,
)
from bowlsense.schemas import FramePose, PhaseEvents, PoseSequence
from synth.skeleton_gen import SynthParams, SynthResult, generate, render_to_video

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)
FPS = 120


@cache
def _settings() -> Settings:
    return load_settings()


def _assert_browser_ready(path: Path) -> None:
    """What Chrome needs: H.264 video, and the index (moov) before the data (mdat)."""
    assert probe_codec(path) == "h264"
    boxes = mp4_top_level_boxes(path)
    assert boxes.index("moov") < boxes.index("mdat"), boxes


def _three_balls() -> tuple[PoseSequence, list[SynthResult], list[int]]:
    """Three synthetic deliveries with 2 s standing still between them (walking back)."""
    specs = [(8.0, 8.0), (22.0, 8.0), (8.0, 30.0)]  # (elbow extension, knee collapse)
    parts = [
        generate(
            SynthParams(
                fps=FPS, height_px=300, seed=k, elbow_extension_deg=e, knee_flex_change_deg=kc
            )
        )
        for k, (e, kc) in enumerate(specs)
    ]
    frames: list[FramePose] = []
    releases = []
    for j, r in enumerate(parts):
        src = list(r.sequence.frames)
        if j < len(parts) - 1:
            src += [src[-1]] * (2 * FPS)
        off = len(frames)
        releases.append(off + r.phases.release)
        frames += [
            FramePose(frame_idx=off + i, t=(off + i) / FPS, keypoints=f.keypoints)
            for i, f in enumerate(src)
        ]
    return parts[0].sequence.model_copy(update={"frames": frames}), parts, releases


@pytest.fixture(scope="module")
def three_ball_clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[int]]:
    d = tmp_path_factory.mktemp("three")
    seq, _, releases = _three_balls()
    video = render_to_video(seq, d / "three.mp4")
    pose = d / "three.json"
    pose.write_text(seq.model_dump_json(), encoding="utf-8")
    return video, pose, releases


@pytest.fixture(scope="module")
def one_ball(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SynthResult]:
    d = tmp_path_factory.mktemp("one")
    r = generate(SynthParams(fps=FPS, height_px=300, elbow_extension_deg=22.0))
    return render_to_video(r.sequence, d / "one.mp4"), r


# --- H.264 writer ------------------------------------------------------------------------------


def test_h264_writer_browser_ready(tmp_path: Path) -> None:
    out = tmp_path / "w.mp4"
    with H264Writer(out, 64, 48, 30.0, "veryfast", 23) as w:
        for i in range(15):
            w.write(np.full((48, 64, 3), i * 10, np.uint8))
    _assert_browser_ready(out)
    info = probe_video(out)
    assert (info.width, info.height, info.frame_count) == (64, 48, 15)
    assert info.fps == pytest.approx(30.0)


def test_h264_writer_pads_odd_sizes(tmp_path: Path) -> None:
    assert even_size(65, 47) == (66, 48)
    out = tmp_path / "odd.mp4"
    with H264Writer(out, 65, 47, 25.0, "veryfast", 23) as w:
        w.write(np.zeros((47, 65, 3), np.uint8))
    assert (probe_video(out).width, probe_video(out).height) == (66, 48)


def test_h264_writer_errors(tmp_path: Path) -> None:
    out = tmp_path / "bad.mp4"
    with (
        pytest.raises(RenderError, match="Frame is"),
        H264Writer(out, 64, 48, 30, "veryfast", 23) as w,
    ):
        w.write(np.zeros((10, 10, 3), np.uint8))
    assert not out.exists()  # partial file removed on error
    with pytest.raises(RenderError, match="No frames"), H264Writer(out, 64, 48, 30, "veryfast", 23):
        pass
    with pytest.raises(RenderError, match="Bad video size"):
        H264Writer(out, 0, 48, 30, "veryfast", 23)


# --- overlay ------------------------------------------------------------------------------


def test_phase_labels() -> None:
    p = PhaseEvents(run_up_start=10, bound=50, bfc=60, ffc=70, arm_horizontal=75, release=80,
                    follow_through_end=120, confidence=1.0)  # fmt: skip
    assert phase_label(5, p) == ""
    assert phase_label(10, p) == "Run-up start"
    assert phase_label(30, p) == "Run-up"
    assert phase_label(55, p) == "Bound"
    assert phase_label(65, p) == "Delivery stride"
    assert phase_label(72, p) == "Arm circle"
    assert phase_label(80, p) == "Release"
    assert phase_label(100, p) == "Follow-through"
    assert phase_label(200, p) == ""


def test_conf_colours() -> None:
    r = _settings().render
    assert conf_colour(0.1, r) == RED and conf_colour(0.5, r) == AMBER
    assert conf_colour(0.9, r) == GREEN


def test_draw_pose_tint_and_banner() -> None:
    r = _settings().render
    frame = np.zeros((120, 160, 3), np.uint8)
    plain = draw_pose(frame, None, r)
    assert np.array_equal(plain, frame) and plain is not frame
    tinted = draw_pose(frame, None, r, tint=True)
    assert tinted[100, 100, 2] > tinted[100, 100, 0]  # red channel up (BGR)
    banner = draw_pose(frame, None, r, label="Release")
    assert banner[5:30, 5:150].max() > 200  # white text on the banner


def test_overlay_video(one_ball: tuple[Path, SynthResult], tmp_path: Path) -> None:
    video, r = one_ball
    s = _settings()
    a = analyze_delivery(r.sequence, s)
    seq = clean_sequence(r.sequence, s.pose)
    seg = OverlaySegment(start=0, stop=len(seq.frames), phases=a.phases, legality=a.legality)
    out = render_overlay(video, seq, [seg], tmp_path / "overlay.mp4", s.render)
    _assert_browser_ready(out)
    assert probe_video(out).frame_count == probe_video(video).frame_count
    # flagged delivery: frames between arm-horizontal and release are tinted red
    inside = read_frame(out, a.phases.arm_horizontal + 1)[200:, :50].mean(axis=(0, 1))
    outside = read_frame(out, a.phases.bfc)[200:, :50].mean(axis=(0, 1))
    assert inside[2] > outside[2] + 20


def test_keyframes(one_ball: tuple[Path, SynthResult], tmp_path: Path) -> None:
    video, r = one_ball
    s = _settings()
    a = analyze_delivery(r.sequence, s)
    seq = clean_sequence(r.sequence, s.pose)
    seg = OverlaySegment(start=0, stop=len(seq.frames), phases=a.phases, legality=a.legality)
    painter = OverlayPainter(seq, [seg], s.render)
    paths = export_keyframes(video, painter, seg, tmp_path / "kf", s.render)
    assert list(paths) == list(s.render.keyframe_events)
    for p in paths.values():
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (seq.height, seq.width)
    assert paths["release"].name == "06_release.png"


# --- clipping ------------------------------------------------------------------------------------


def test_gate_clipping_finds_three_balls() -> None:
    """Phase 8 gate: clipping splits a synthetic 3-ball video into 3."""
    seq, _, truth = _three_balls()
    s = _settings()
    found = find_releases(clean_sequence(seq, s.pose), s.clipping)
    assert len(found) == 3
    assert all(abs(f - t) <= 2 for f, t in zip(found, truth, strict=True))
    windows = clip_windows(found, len(seq.frames), FPS, s.clipping)
    for w, t in zip(windows, truth, strict=True):
        assert w.start <= t < w.stop
        assert w.release - w.start == round(s.clipping.before_release_s * FPS)
    assert windows[-1].stop <= len(seq.frames)


def test_find_releases_single_and_errors() -> None:
    s = _settings()
    r = generate(SynthParams(fps=FPS))
    assert find_releases(clean_sequence(r.sequence, s.pose), s.clipping) == [
        pytest.approx(r.phases.release, abs=2)
    ]
    frames = [
        f.model_copy(update={"keypoints": {**f.keypoints, "r_wrist": None}})
        for f in r.sequence.frames
    ]
    with pytest.raises(PhaseDetectionError):
        find_releases(r.sequence.model_copy(update={"frames": frames}), s.clipping)


def test_clip_windows_clamp() -> None:
    s = _settings().clipping
    w = clip_windows([10, 990], 1000, 100.0, s)
    assert w[0].start == 0 and w[1].stop == 1000


def test_sub_sequence_and_cut(
    three_ball_clip: tuple[Path, Path, list[int]], tmp_path: Path
) -> None:
    video, pose, _ = three_ball_clip
    seq = PoseSequence.model_validate_json(pose.read_text(encoding="utf-8"))
    window = ClipWindow(start=100, stop=160, release=130)
    sub = sub_sequence(seq, window)
    assert [f.frame_idx for f in sub.frames] == list(range(60))
    assert sub.frames[0].keypoints == seq.frames[100].keypoints
    clip = cut_clip(video, window, tmp_path / "clip.mp4", _settings().render)
    _assert_browser_ready(clip)
    assert probe_video(clip).frame_count == 60


# --- PDF ----------------------------------------------------------------------------------------


def _report(one_ball: tuple[Path, SynthResult], tmp_path: Path, coach: str | None) -> Path:
    video, r = one_ball
    s = _settings()
    a = analyze_delivery(r.sequence, s)
    seq = clean_sequence(r.sequence, s.pose)
    seg = OverlaySegment(start=0, stop=len(seq.frames), phases=a.phases, legality=a.legality)
    kf = export_keyframes(video, OverlayPainter(seq, [seg], s.render), seg, tmp_path / "kf",
                          s.render)  # fmt: skip
    data = ReportInput(
        video_name="one.mp4",
        created=datetime(2026, 11, 20, 9, 0),
        view="side",
        bowling_arm="right",
        fps=FPS,
        quality_warnings=["Resolution is below 720p"],
        deliveries=[ReportDelivery(index=0, analysis=a, sequence=seq, keyframes=kf)],
        coach_text=coach,
    )
    return build_report(data, tmp_path / "r.pdf", tmp_path / "charts", 15.0)


def test_gate_pdf_opens_with_disclaimers(
    one_ball: tuple[Path, SynthResult], tmp_path: Path
) -> None:
    """Phase 8 gate: the PDF opens; it carries both disclaimers and the analysis."""
    pdf = _report(one_ball, tmp_path, coach=None)
    assert pdf.read_bytes()[:5] == b"%PDF-"
    reader = pypdf.PdfReader(str(pdf))
    assert len(reader.pages) >= 2
    text = " ".join(page.extract_text() for page in reader.pages).replace("\n", " ")
    assert "2D video screening only" in text and "not medical advice" in text
    assert "FLAGGED" in text and "Resolution is below 720p" in text
    assert "Coach's notes" not in text
    assert (tmp_path / "charts" / "delivery_01_angles.png").is_file()
    assert LEGALITY_NOTE.startswith("2D") and "not medical advice" in INJURY_NOTE


def test_pdf_includes_coach_text(one_ball: tuple[Path, SynthResult], tmp_path: Path) -> None:
    pdf = _report(one_ball, tmp_path, coach="Keep your front leg firm.\nWork on your run-up.")
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(str(pdf)).pages)
    assert "Coach's notes" in text and "Keep your front leg firm." in text


# --- CLI end to end -----------------------------------------------------------------------------


def test_gate_cli_three_ball_session(
    three_ball_clip: tuple[Path, Path, list[int]], tmp_path: Path
) -> None:
    video, pose, releases = three_ball_clip
    out = tmp_path / "out"
    assert cli_main(["analyze", str(video), "--out", str(out), "--pose-from-json", str(pose)]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["view"] == "side" and len(result["deliveries"]) == 3
    balls = result["deliveries"]
    assert all(abs(b["clip"]["release"] - t) <= 2 for b, t in zip(balls, releases, strict=True))
    assert [b["legality"]["flagged"] for b in balls] == [False, True, False]
    assert "KNEE_COLLAPSE" in {f["code"] for f in balls[2]["faults"]["faults"]}
    assert all(b["phases"]["confidence"] == 1.0 for b in balls)
    _assert_browser_ready(out / "overlay.mp4")
    assert len(pypdf.PdfReader(str(out / "report.pdf")).pages) >= 4
    assert len(list((out / "keyframes").rglob("*.png"))) == 15
