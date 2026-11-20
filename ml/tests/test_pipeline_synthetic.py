"""Phase 12 gate: the full session pipeline on a rendered synthetic 3-ball video."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pypdf
import pytest

from bowlsense import offline
from bowlsense.config import Settings, load_settings
from bowlsense.pipeline import PipelineOptions, run_pipeline
from bowlsense.schemas import SessionResult
from synth.sessions import multi_ball_session
from synth.skeleton_gen import SynthParams, render_to_video

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)
FPS = 120


@pytest.fixture(scope="module")
def session_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, list[int]]:
    """Three deliveries: clean, illegal elbow, knee collapse (300 px tall, 120 fps)."""
    d = tmp_path_factory.mktemp("session")
    seq, _, releases = multi_ball_session(
        [
            SynthParams(fps=FPS, height_px=300, seed=0),
            SynthParams(fps=FPS, height_px=300, seed=1, elbow_extension_deg=22.0),
            SynthParams(fps=FPS, height_px=300, seed=2, knee_flex_change_deg=30.0),
        ]
    )
    video = render_to_video(seq, d / "three.mp4")
    pose = d / "pose.json"
    pose.write_text(seq.model_dump_json(), encoding="utf-8")
    return video, pose, releases


@pytest.fixture(scope="module")
def run(
    session_video: tuple[Path, Path, list[int]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[SessionResult, Path, list[tuple[float, str]]]:
    video, pose, _ = session_video
    out = tmp_path_factory.mktemp("out")
    steps: list[tuple[float, str]] = []
    history = [(date(2026, 10, d), 30) for d in range(1, 29)]
    options = PipelineOptions(
        pose_json=pose, use_llm=False, recorded_on=date(2026, 10, 29), workload_history=history
    )
    was_on = offline.is_enabled()
    offline.enable()  # the whole pipeline must work with the network blocked
    try:
        result = run_pipeline(
            video, out, load_settings(), session_id="s1", options=options,
            progress=lambda p, s: steps.append((p, s)),
        )  # fmt: skip
    finally:
        if not was_on:
            offline.disable()
    return result, out, steps


def test_gate_session_result(
    run: tuple[SessionResult, Path, list[tuple[float, str]]],
    session_video: tuple[Path, Path, list[int]],
) -> None:
    result, _, _ = run
    _, _, releases = session_video
    assert result.session_id == "s1" and len(result.deliveries) == 3
    assert [d.delivery_idx for d in result.deliveries] == [0, 1, 2]
    for d, truth in zip(result.deliveries, releases, strict=True):
        assert abs(d.clip_start + d.phases.release - truth) <= 2
    assert [d.legality.flagged for d in result.deliveries] == [False, True, False]
    assert "KNEE_COLLAPSE" in {f.code for f in result.deliveries[2].faults}
    assert result.view == "side" and result.bowling_arm == "right"
    assert result.pose_backend == "pose-json" and result.fps == pytest.approx(FPS)
    assert result.consistency_score is not None and 0 <= result.consistency_score <= 100
    assert result.coach_source == "template" and "not medical advice" in result.coach_report


def test_delivery_details(run: tuple[SessionResult, Path, list[tuple[float, str]]]) -> None:
    result, out, _ = run
    d = result.deliveries[2]
    names = [a.name for a in d.angles]
    assert names == ["elbow", "front_knee", "trunk_lean"]
    elbow = d.angles[0]
    assert d.clip_stop is not None
    assert len(elbow.values) == len(elbow.frames) == d.clip_stop - d.clip_start
    assert d.stride_pct is not None and 40 < d.stride_pct < 120
    assert d.release_height_pct is not None and d.release_height_pct > 100
    assert d.alignment is not None and d.action_type == d.alignment.action_type
    assert d.comparisons and d.comparisons[0].similarity_pct >= d.comparisons[-1].similarity_pct
    assert set(d.keyframes) >= {"bfc", "ffc", "release"}
    assert all((out / p).is_file() for p in d.keyframes.values())
    assert d.speed_kmh is None  # no calibration: speed is never guessed


def test_session_level_modules(run: tuple[SessionResult, Path, list[tuple[float, str]]]) -> None:
    result, _, _ = run
    sims = [s for _, s in result.similar_pros]
    assert sims == sorted(sims, reverse=True) and result.similar_pros[0][0].startswith("Reference")
    assert result.workload.balls_this_session == 3
    assert result.workload.acwr is not None  # 28 days of history
    assert result.fatigue.fatigue_index >= 0


def test_artifacts_written(run: tuple[SessionResult, Path, list[tuple[float, str]]]) -> None:
    result, out, steps = run
    for key in ("overlay", "report", "pose", "coach_summary"):
        assert (out / result.artifacts[key]).is_file(), key
    saved = SessionResult.model_validate_json((out / "result.json").read_text(encoding="utf-8"))
    assert saved == result
    assert json.loads((out / "coach_summary.json").read_text(encoding="utf-8"))["deliveries"] == 3
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(str(out / "report.pdf")).pages)
    assert "Coach's notes" in text
    pcts = [p for p, _ in steps]
    assert pcts == sorted(pcts) and pcts[-1] == 100 and steps[-1][1] == "done"


def test_no_profile_means_no_workload(
    session_video: tuple[Path, Path, list[int]], tmp_path: Path, settings: Settings
) -> None:
    """Single delivery, no bowler profile: no ACWR and no consistency score."""
    video, pose, releases = session_video
    seq = json.loads(pose.read_text(encoding="utf-8"))
    seq["frames"] = seq["frames"][: releases[0] + 200]
    one = tmp_path / "one.json"
    one.write_text(json.dumps(seq), encoding="utf-8")
    result = run_pipeline(
        video, tmp_path / "o", settings, session_id="s2",
        options=PipelineOptions(pose_json=one, use_llm=False), track_ball_flight=False,
    )  # fmt: skip
    assert len(result.deliveries) == 1 and result.consistency_score is None
    assert result.workload.acwr is None and "No bowler profile" in result.workload.note
