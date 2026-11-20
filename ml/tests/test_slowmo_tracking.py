"""Slow-motion clips, identity swaps and broken tracks (real-footage failure modes)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from bowlsense.config import load_settings
from bowlsense.detect.person import (
    PersonTrack,
    colour_distance,
    split_on_colour,
    stitch_for,
    stitch_tracks,
)
from bowlsense.io.slowmo import estimate_slowmo, retime
from bowlsense.pose.smoothing import clean_sequence
from synth.skeleton_gen import SynthParams, generate, render_to_video

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)

BLUE = [1.0] + [0.0] * 47
GREEN = [0.0] * 20 + [1.0] + [0.0] * 27


def _track(tid: int, frames: range, x0: float, dx: float, colour: list[float]) -> PersonTrack:
    fs = list(frames)
    boxes = [(x0 + dx * i, 100.0, x0 + dx * i + 60.0, 300.0) for i in range(len(fs))]
    return PersonTrack(track_id=tid, frames=fs, boxes=boxes, confs=[0.9] * len(fs),
                       colours=[colour] * len(fs))  # fmt: skip


# --- slow motion -------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [4.0, 8.0])
def test_slow_motion_detected_and_undone(factor: float) -> None:
    s = load_settings()
    r = generate(SynthParams(fps=240.0))
    real = clean_sequence(r.sequence, s.pose)
    assert estimate_slowmo(real, s).factor == 1.0  # real-time footage is left alone
    slowed = retime(r.sequence, 1.0 / factor)  # same frames, labelled 240/factor fps
    est = estimate_slowmo(clean_sequence(slowed, s.pose), s)
    assert est.source == "auto" and est.factor == factor
    back = retime(slowed, est.factor)
    assert back.fps == pytest.approx(240.0)
    assert back.frames[100].t == pytest.approx(r.sequence.frames[100].t)


def test_slow_motion_through_the_pipeline(tmp_path: Path) -> None:
    """An 8x slowed side-on clip is analysed at its real frame rate and judged (8 deg: legal);
    a user-given factor and camera view are used as given."""
    from bowlsense.pipeline import extract_pose

    s = load_settings()
    r = generate(SynthParams(fps=240.0, height_px=300.0, elbow_extension_deg=8.0))
    slowed = retime(r.sequence, 1.0 / 8.0)  # labelled 30 fps
    video = render_to_video(slowed, tmp_path / "slow.mp4")
    pose = tmp_path / "pose.json"
    pose.write_text(slowed.model_dump_json(), encoding="utf-8")
    front = extract_pose(video, s, pose_json=pose, slow_motion=None)
    assert front.slow_motion.factor == 8.0 and front.video.fps == pytest.approx(240.0)
    assert front.video_fps == pytest.approx(30.0) and "Slow-motion" in front.warnings[0]
    given = extract_pose(video, s, pose_json=pose, slow_motion=8.0, camera_view="behind")
    assert given.slow_motion.source == "user" and given.sequence.view == "behind"
    from bowlsense.pipeline import analyze_delivery

    d = analyze_delivery(front.sequence, s)
    assert d.legality.screened and d.legality.verdict == "legal"


# --- tracking: colour splits and stitching -----------------------------------------------------


def test_colour_distance() -> None:
    assert colour_distance(BLUE, BLUE) == pytest.approx(0.0)
    assert colour_distance(BLUE, GREEN) == pytest.approx(1.0)


def test_identity_swap_is_split_and_restitched() -> None:
    """The tracker swaps the bowler (blue) and batter (green) when they cross: colours put
    each person back together."""
    s = load_settings()
    # track 1: batter frames 0-99, then (after the crossing) the bowler frames 100-199
    swapped = PersonTrack(
        track_id=1, frames=list(range(200)), confs=[0.9] * 200,
        boxes=[(500.0, 100.0, 560.0, 300.0)] * 100 + [(510.0, 100.0, 570.0, 300.0)] * 100,
        colours=[GREEN] * 100 + [BLUE] * 100,
    )  # fmt: skip
    bowler_before = _track(5, range(0, 95), 100.0, 4.2, BLUE)  # runs up to the crossing
    batter_after = _track(9, range(105, 200), 505.0, 0.0, GREEN)  # batter gets a new id
    pieces = split_on_colour([swapped], window=10, threshold=0.4, min_piece=10)
    assert sorted(len(p.frames) for p in pieces) == [100, 100]
    joined = stitch_for([swapped, bowler_before, batter_after], s.detect, fps=30.0)
    colours = {t.track_id: np.mean(t.colours, axis=0) for t in joined}
    by_len = {t.track_id: (t.frames[0], t.frames[-1]) for t in joined}
    bowler = next(t for t in joined if colour_distance(colours[t.track_id], BLUE) < 0.2)
    batter = next(t for t in joined if colour_distance(colours[t.track_id], GREEN) < 0.2)
    assert by_len[bowler.track_id] == (0, 199)
    assert batter.frames[0] == 0 and batter.frames[-1] == 199


def test_stitch_needs_position_size_time_and_colour() -> None:
    a = _track(1, range(0, 50), 100.0, 2.0, BLUE)
    near = _track(2, range(60, 90), 200.0, 2.0, BLUE)  # continues a
    far = _track(3, range(60, 90), 900.0, 2.0, BLUE)  # too far away
    late = _track(4, range(200, 230), 200.0, 2.0, BLUE)  # gap too long
    other = _track(5, range(60, 90), 200.0, 2.0, GREEN)  # someone else
    out = stitch_tracks([a, far, late, near], max_gap_frames=30, max_colour_distance=0.4)
    assert {t.track_id: len(t.frames) for t in out} == {1: 80, 3: 30, 4: 30}
    out = stitch_tracks([a, other], max_gap_frames=30, max_colour_distance=0.4)
    assert {t.track_id: len(t.frames) for t in out} == {1: 50, 5: 30}
