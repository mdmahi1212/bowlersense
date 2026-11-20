"""Tests for the real-video front end: video I/O, quality checks, bowler selection, pose
backends + keypoint mapping, the pipeline front end and the CLI (Phase 7)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from bowlsense import offline
from bowlsense.cli import main as cli_main
from bowlsense.config import Settings, load_settings
from bowlsense.detect.person import (
    PersonTrack,
    model_path,
    select_bowler,
    track_motion,
    track_people,
)
from bowlsense.errors import (
    InvalidInputError,
    LowFpsError,
    ModelNotFoundError,
    NoBowlerDetectedError,
    UnsupportedMediaError,
)
from bowlsense.geometry.body import runup_window
from bowlsense.io.quality import check_pose, check_video, estimate_view
from bowlsense.io.video import (
    VideoInfo,
    is_image,
    iter_frames,
    load_image,
    probe_video,
    read_frame,
)
from bowlsense.pipeline import analyze_delivery, extract_pose
from bowlsense.pose import base as pose_base
from bowlsense.pose.base import KeypointMap, load_backend, pad_bbox
from bowlsense.pose.mediapipe_backend import MEDIAPIPE_TO_BOWLSENSE, map_mediapipe
from bowlsense.pose.rtm import HALPE26_TO_BOWLSENSE, map_halpe26
from bowlsense.schemas import KEYPOINT_NAMES, PoseSequence
from synth.skeleton_gen import SynthParams, SynthResult, generate, render_to_video
from tests.conftest import real_clips

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)
MODELS_PRESENT = all(
    (load_settings().storage.models_path / name).is_file()
    for name in ("yolo11n.pt", "rtmpose-m_halpe26_256x192.onnx", "pose_landmarker_full.task")
)
needs_models = pytest.mark.skipif(
    not MODELS_PRESENT, reason="model weights not downloaded (scripts/download_models.py)"
)


@cache
def _settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="module")
def synth_clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, SynthResult]:
    """A rendered synthetic delivery (60 fps to keep it small) plus its pose JSON."""
    d = tmp_path_factory.mktemp("clip")
    r = generate(SynthParams(fps=120, height_px=360, elbow_extension_deg=22.0))
    video = render_to_video(r.sequence, d / "synth.mp4")
    pose = d / "pose.json"
    pose.write_text(r.sequence.model_dump_json(), encoding="utf-8")
    return video, pose, r


# --- video I/O -----------------------------------------------------------------------------------


def test_probe_and_iterate(synth_clip: tuple[Path, Path, SynthResult]) -> None:
    video, _, r = synth_clip
    info = probe_video(video)
    assert info.fps == pytest.approx(120.0)
    assert (info.width, info.height) == (r.sequence.width, r.sequence.height)
    assert info.frame_count == len(r.sequence.frames)
    assert info.duration_s == pytest.approx(len(r.sequence.frames) / 120.0)
    frames = list(iter_frames(video, start=10, stop=13))
    assert [i for i, _ in frames] == [10, 11, 12]
    assert frames[0][1].shape == (info.height, info.width, 3)
    assert np.array_equal(read_frame(video, 11), frames[1][1])
    with pytest.raises(InvalidInputError, match="beyond the end"):
        read_frame(video, info.frame_count + 5)


def test_video_errors(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedMediaError):
        probe_video(tmp_path / "clip.gif")
    with pytest.raises(InvalidInputError, match="not found"):
        probe_video(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(InvalidInputError):
        probe_video(junk)
    with pytest.raises(InvalidInputError):
        list(iter_frames(junk, start=-1))


def test_images(tmp_path: Path) -> None:
    img = np.zeros((20, 30, 3), np.uint8)
    img[5:10, 5:10] = 255
    path = tmp_path / "pic.png"
    cv2.imwrite(str(path), img)
    assert is_image(path) and not is_image(tmp_path / "x.mp4")
    assert np.array_equal(load_image(path), img)
    with pytest.raises(UnsupportedMediaError):
        load_image(tmp_path / "pic.bmp")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(InvalidInputError):
        load_image(bad)


# --- quality checks ------------------------------------------------------------------------------


def _info(fps: float, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(path="x.mp4", fps=fps, width=width, height=height, frame_count=100)


def test_check_video_fps_and_resolution() -> None:
    v = _settings().video
    assert check_video(_info(240), v) == []
    assert any("release metrics less reliable" in w for w in check_video(_info(100), v))
    with pytest.raises(LowFpsError):
        check_video(_info(30), v)
    assert any("below the 60" in w for w in check_video(_info(30), v, allow_low_fps=True))
    assert any("720p" in w for w in check_video(_info(240, 1280, 600), v))


def test_check_pose_visibility_and_confidence() -> None:
    v = _settings().video
    r = generate(SynthParams(fps=60))
    n = len(r.sequence.frames)
    assert check_pose(r.sequence, n, v) == []
    assert any("only 50%" in w for w in check_pose(r.sequence, n // 2, v))
    frames = [
        f.model_copy(
            update={
                "keypoints": {
                    k: (kp.model_copy(update={"conf": 0.3}) if kp else None)
                    for k, kp in f.keypoints.items()
                }
            }
        )
        for f in r.sequence.frames
    ]
    low = r.sequence.model_copy(update={"frames": frames})
    assert any("confidence" in w for w in check_pose(low, n, v))


@pytest.mark.parametrize("view", ["side", "front", "behind"])
@pytest.mark.parametrize("arm", ["right", "left"])
def test_estimate_view(view: str, arm: str) -> None:
    s = _settings()
    seq = generate(
        SynthParams.model_validate({"view": view, "bowling_arm": arm, "noise_px": 2.0})
    ).sequence
    assert estimate_view(seq, runup_window(seq, s.body), s.video) == view


def test_estimate_view_unknown_when_empty() -> None:
    seq = generate(SynthParams(fps=60)).sequence
    assert estimate_view(seq, (0, 1), _settings().video) == "unknown"


# --- bowler selection ----------------------------------------------------------------------------


def _track(tid: int, xs: list[float], start: int = 0) -> PersonTrack:
    frames = list(range(start, start + len(xs)))
    boxes = [(x - 20.0, 100.0, x + 20.0, 300.0) for x in xs]
    return PersonTrack(track_id=tid, frames=frames, boxes=boxes, confs=[0.9] * len(xs))


def test_select_bowler_prefers_runner() -> None:
    umpire = _track(1, [900.0 + (i % 3) for i in range(200)])  # standing still
    bowler = _track(7, [100.0 + 6.0 * i for i in range(150)])  # running across
    fielder = _track(3, [500.0 + 1.0 * i for i in range(200)])  # walking
    short = _track(9, [0.0, 400.0, 800.0])  # too short to count
    chosen = select_bowler([umpire, bowler, fielder, short], 1920, min_frames=10)
    assert chosen.track_id == 7


def test_learned_selector_scores_and_picks() -> None:
    """The shipped selector loads, scores every usable track and prefers the runner over a
    standing umpire and a slow walker."""
    from bowlsense.detect.selector import FEATURES, feature_matrix, load_selector

    s = _settings()
    path = s.storage.models_path / s.detect.selector_model
    if not path.is_file():
        pytest.skip("bowler_selector.json not trained")
    umpire = _track(1, [900.0 + (i % 3) for i in range(200)])
    bowler = _track(7, [100.0 + 6.0 * i for i in range(150)])
    fielder = _track(3, [500.0 + 1.0 * i for i in range(200)])
    tracks = [umpire, bowler, fielder]
    assert feature_matrix(tracks, 1920, 1080, 200).shape == (3, len(FEATURES))
    scorer = load_selector(path)
    scores = scorer.scores(tracks, 1920, 1080, 200)
    assert scores.shape == (3,) and ((scores >= 0) & (scores <= 1)).all()
    chosen = select_bowler(tracks, 1920, 10, scorer=scorer, frame_height=1080, n_frames=200)
    assert chosen.track_id == 7


def test_selector_rejects_other_features(tmp_path: Path) -> None:
    import json

    from bowlsense.detect.selector import load_selector

    bad = tmp_path / "sel.json"
    bad.write_text(json.dumps({"feature_version": 0, "features": ["x"], "mean": [0],
                               "scale": [1], "coef": [1], "intercept": 0}))  # fmt: skip
    with pytest.raises(ValueError, match="retrain"):
        load_selector(bad)


def test_select_bowler_override_and_errors() -> None:
    a, b = _track(1, [0.0] * 20), _track(2, [float(i) for i in range(20)])
    assert select_bowler([a, b], 1920, 10, override_id=1).track_id == 1
    with pytest.raises(NoBowlerDetectedError, match="tracks in this clip: 1, 2"):
        select_bowler([a, b], 1920, 10, override_id=5)
    with pytest.raises(NoBowlerDetectedError, match="at least 10 frames"):
        select_bowler([_track(1, [0.0, 1.0])], 1920, 10)
    with pytest.raises(NoBowlerDetectedError):
        select_bowler([], 1920, 10)


def test_track_motion_and_box_at() -> None:
    t = _track(1, [0.0, 10.0, 20.0, 1000.0])  # one ID-switch jump
    disp, peak = track_motion(t, 1000)
    assert disp == pytest.approx(1.0)
    assert peak < 1.0  # the 95th percentile tempers the single jump
    assert t.box_at(2) == (0.0, 100.0, 40.0, 300.0) and t.box_at(99) is None
    assert track_motion(_track(1, [5.0]), 100) == (0.0, 0.0)


# --- pose backends and mapping -----------------------------------------------------------------


def test_halpe26_mapping() -> None:
    kps = np.arange(52, dtype=np.float64).reshape(26, 2)
    scores = np.linspace(0.0, 1.3, 26)  # RTMPose scores can exceed 1
    out = map_halpe26(kps, scores)
    assert set(out) == set(KEYPOINT_NAMES)
    assert all(v is not None for v in out.values())
    for idx, name in HALPE26_TO_BOWLSENSE.items():
        kp = out[name]
        assert kp is not None and (kp.x, kp.y) == (kps[idx, 0], kps[idx, 1])
        assert 0.0 <= kp.conf <= 1.0
    l_heel = out["l_heel"]
    assert l_heel is not None and l_heel.x == 48.0  # Halpe index 24
    with pytest.raises(ValueError):
        map_halpe26(np.zeros((17, 2)), np.zeros(17))


def test_mediapipe_mapping_to_frame_coords() -> None:
    marks = [(i / 40.0, 0.5, 0.9) for i in range(33)]
    out = map_mediapipe(marks, (100.0, 50.0, 300.0, 450.0))  # crop 200 x 400
    nose = out["nose"]
    assert nose is not None and (nose.x, nose.y) == (100.0, 250.0)
    toe = out["r_toe"]
    assert toe is not None and toe.x == pytest.approx(100.0 + 200.0 * 32 / 40.0)
    assert set(MEDIAPIPE_TO_BOWLSENSE.values()) == set(KEYPOINT_NAMES)
    with pytest.raises(ValueError):
        map_mediapipe(marks[:10], (0.0, 0.0, 1.0, 1.0))


def test_pad_bbox() -> None:
    assert pad_bbox((100, 100, 200, 300), 0.2, 1000, 1000) == (80.0, 60.0, 220.0, 340.0)
    assert pad_bbox((5, 5, 95, 95), 0.5, 100, 100) == (0.0, 0.0, 100.0, 100.0)


def test_missing_model_message(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match=r"download_models.py"):
        model_path(tmp_path, "yolo11n.pt")
    with pytest.raises(ModelNotFoundError):
        load_backend("rtm", _settings().pose, tmp_path)


def test_unknown_backend() -> None:
    from bowlsense.errors import ConfigError

    with pytest.raises(ConfigError, match="Unknown pose backend"):
        load_backend("openpose", _settings().pose, _settings().storage.models_path)


@pytest.fixture()
def offline_guard(no_offline_guard: None) -> Iterator[None]:
    """Run the test with every non-loopback network connection blocked."""
    offline.enable()
    yield
    offline.disable()


@needs_models
@pytest.mark.parametrize("name", ["rtm", "mediapipe"])
def test_backends_run_offline(name: str, offline_guard: None) -> None:
    s = _settings()
    backend = load_backend(name, s.pose, s.storage.models_path)
    assert backend.name == name
    out = backend.estimate(np.full((480, 640, 3), 128, np.uint8), (200.0, 40.0, 400.0, 460.0))
    assert set(out) == set(KEYPOINT_NAMES)
    for kp in out.values():
        assert kp is None or (0.0 <= kp.conf <= 1.0)


@needs_models
def test_person_tracker_runs_offline(
    synth_clip: tuple[Path, Path, SynthResult], offline_guard: None
) -> None:
    s = _settings()
    video, _, _ = synth_clip
    weights = model_path(s.storage.models_path, s.detect.yolo_model)
    tracks = track_people(iter_frames(video, 0, 30), weights, s.detect)
    assert isinstance(tracks, list)  # a stick figure may or may not register as a person


# --- pipeline front end and CLI ----------------------------------------------------------------


def test_gate_cli_analyze_with_pose_json(
    synth_clip: tuple[Path, Path, SynthResult], tmp_path: Path
) -> None:
    """Phase 7 gate: `cli analyze` runs on the synthetic video with --pose-from-json."""
    video, pose, r = synth_clip
    out = tmp_path / "out"
    assert (
        cli_main(
            ["analyze", str(video), "--out", str(out), "--pose-from-json", str(pose), "--no-llm"]
        )
        == 0
    )
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["pose_backend"] == "pose-json" and result["bowler_track_id"] is None
    assert result["view"] == "side" and result["bowling_arm"] == "right"
    assert len(result["deliveries"]) == 1
    phases = result["deliveries"][0]["phases"]
    start = result["deliveries"][0]["clip_start"]  # phases are numbered within the clip
    for event in ("bfc", "ffc", "release"):
        assert abs(start + phases[event] - getattr(r.phases, event)) <= 2
    assert result["deliveries"][0]["legality"]["flagged"] is True
    assert "not medical advice" in result["deliveries"][0]["injury"]["disclaimer"]
    saved = PoseSequence.model_validate_json((out / "pose.json").read_text(encoding="utf-8"))
    assert len(saved.frames) == len(r.sequence.frames)


def test_cli_reports_errors_with_exit_2(
    synth_clip: tuple[Path, Path, SynthResult], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video, _, _ = synth_clip
    code = cli_main(["analyze", str(tmp_path / "nope.mp4"), "--out", str(tmp_path)])
    assert code == 2 and "not found" in capsys.readouterr().err
    other = generate(SynthParams(fps=60, height_px=200)).sequence
    bad_pose = tmp_path / "other.json"
    bad_pose.write_text(other.model_dump_json(), encoding="utf-8")
    code = cli_main(
        ["analyze", str(video), "--out", str(tmp_path / "o"), "--pose-from-json", str(bad_pose)]
    )
    assert code == 2 and "Pose JSON" in capsys.readouterr().err


class _GroundTruthWrapper:
    """Runs a REAL backend on every bowler frame (proving it loads and runs), then hands the
    analysis the synthetic ground-truth keypoints, since pose models don't detect stick
    figures."""

    def __init__(self, inner: Any, truth: PoseSequence) -> None:
        self.inner, self.truth, self.calls = inner, truth, 0
        self.name = inner.name

    def estimate(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> KeypointMap:
        self.inner.estimate(frame, bbox)
        self.calls += 1
        return dict(self.truth.frames[self.calls - 1].keypoints)


@needs_models
@pytest.mark.parametrize("backend", ["mediapipe", "rtm"])
def test_backend_switch_runs_real_model_through_pipeline(
    synth_clip: tuple[Path, Path, SynthResult], monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Phase 7 gate: a `--backend mediapipe` switch works through the real-video path."""
    video, _, r = synth_clip
    s = _settings()
    n = len(r.sequence.frames)
    boxes = []
    for f in r.sequence.frames:
        xs = [kp.x for kp in f.keypoints.values() if kp is not None]
        ys = [kp.y for kp in f.keypoints.values() if kp is not None]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    track = PersonTrack(track_id=4, frames=list(range(n)), boxes=boxes, confs=[0.9] * n)
    monkeypatch.setattr("bowlsense.pipeline.track_people_in_video", lambda *a, **k: [track])
    wrappers: list[_GroundTruthWrapper] = []

    def load(name: str, settings: Any, models_dir: Any) -> _GroundTruthWrapper:
        w = _GroundTruthWrapper(pose_base.load_backend(name, settings, models_dir), r.sequence)
        wrappers.append(w)
        return w

    monkeypatch.setattr("bowlsense.pipeline.load_backend", load)
    front = extract_pose(video, s, backend=backend)
    assert front.backend == backend and front.bowler_track_id == 4
    assert wrappers[0].name == backend and wrappers[0].calls == n
    delivery = analyze_delivery(front.sequence, s)
    assert abs(delivery.phases.release - r.phases.release) <= 2


@pytest.mark.skipif(not real_clips(), reason="no real clips in tests/data/real")
@needs_models
@pytest.mark.parametrize("clip", real_clips(), ids=lambda p: p.name)
def test_real_clip_end_to_end(clip: Path, tmp_path: Path) -> None:
    code = cli_main(["analyze", str(clip), "--out", str(tmp_path), "--allow-low-fps"])
    assert code == 0
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["deliveries"][0]["phases"]["release"] > 0
