"""Tests for geometry/body.py: height, bowling arm, role resolution, mirroring."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from bowlsense.config import Settings, load_settings
from bowlsense.detect.person import PersonTrack, select_bowler
from bowlsense.errors import InvalidInputError, PoseQualityError
from bowlsense.geometry.angles import keypoint_track
from bowlsense.geometry.body import (
    BowlingArmResult,
    detect_bowling_arm,
    estimate_height_px,
    has_runup,
    joint,
    mirror_sequence,
    runup_window,
    side_of,
    to_pct,
    with_bowling_arm,
)
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import FramePose, Keypoint, PoseSequence
from synth.skeleton_gen import SynthParams, generate


@pytest.mark.parametrize(
    ("arm", "role", "part", "expected"),
    [
        ("right", "bowling", "wrist", "r_wrist"),
        ("right", "front", "ankle", "l_ankle"),
        ("right", "back", "ankle", "r_ankle"),
        ("right", "non_bowling", "wrist", "l_wrist"),
        ("left", "bowling", "wrist", "l_wrist"),
        ("left", "front", "knee", "r_knee"),
        ("left", "back", "heel", "l_heel"),
    ],
)
def test_role_resolution(arm: str, role: str, part: str, expected: str) -> None:
    assert joint(arm, role, part) == expected  # type: ignore[arg-type]


def test_role_resolution_rejects_unpaired_joint() -> None:
    with pytest.raises(InvalidInputError):
        joint("right", "front", "nose")
    assert side_of("left", "front") == "r"


def test_to_pct() -> None:
    np.testing.assert_allclose(to_pct([150.0, 300.0], 300.0), [50.0, 100.0])
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(InvalidInputError):
            to_pct(1.0, bad)


ARM_CASES = list(
    itertools.product(
        ("right", "left"),
        ("side", "front", "behind"),
        ((0.0, 0.0), (2.0, 0.0), (2.0, 0.1), (4.0, 0.1)),
        (0, 1, 2),
    )
)


@pytest.mark.parametrize(("arm", "view", "noise", "seed"), ARM_CASES)
def test_bowling_arm_detection_100pct(
    settings: Settings, arm: str, view: str, noise: tuple[float, float], seed: int
) -> None:
    """Phase 3 gate: left/right arm detection 100% on synthetic data."""
    p = SynthParams(
        bowling_arm=arm,
        view=view,
        noise_px=noise[0],
        dropout_rate=noise[1],
        seed=seed,
    )
    seq = clean_sequence(generate(p).sequence, settings.pose)
    result = detect_bowling_arm(seq)
    assert result.arm == arm
    assert result.ratio > 1.3  # clear margin, not just a coin flip


def test_with_bowling_arm_sets_field(settings: Settings) -> None:
    seq = generate(SynthParams(bowling_arm="left")).sequence.model_copy(
        update={"bowling_arm": "right"}
    )
    assert with_bowling_arm(seq).bowling_arm == "left"


def test_bowling_arm_ratio_inf_when_one_static() -> None:
    r = BowlingArmResult(arm="right", left_peak_px_s=0.0, right_peak_px_s=10.0)
    assert r.ratio == float("inf")


def _seq_without(name_prefix: str) -> PoseSequence:
    seq = generate(SynthParams(fps=60)).sequence
    frames = [
        f.model_copy(
            update={
                "keypoints": {
                    k: (None if k.startswith(name_prefix) else v) for k, v in f.keypoints.items()
                }
            }
        )
        for f in seq.frames
    ]
    return seq.model_copy(update={"frames": frames})


def test_one_hidden_wrist_picks_the_visible_arm() -> None:
    r = detect_bowling_arm(_seq_without("l_wrist"))
    assert r.arm == "right" and r.method == "only_visible_wrist"
    r = detect_bowling_arm(_seq_without("r_wrist"))
    assert r.arm == "left" and r.method == "only_visible_wrist"


def test_no_wrists_at_all_raises() -> None:
    seq = _seq_without("l_wrist")
    seq = seq.model_copy(update={"frames": [
        f.model_copy(update={"keypoints": {**f.keypoints, "r_wrist": None}}) for f in seq.frames
    ]})  # fmt: skip
    with pytest.raises(PoseQualityError, match="Neither wrist"):
        detect_bowling_arm(seq)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"view": "front"},
        {"view": "behind"},
        {"bowling_arm": "left"},
        {"height_px": 500},
        {"fps": 240},
        {"noise_px": 2.0, "dropout_rate": 0.1, "seed": 3},
        {"stride_pct": 95, "action_type": "mixed"},
    ],
)
def test_height_estimate_within_2pct(settings: Settings, kw: dict[str, object]) -> None:
    r = generate(SynthParams.model_validate(kw))
    seq = clean_sequence(r.sequence, settings.pose)
    est = estimate_height_px(seq, settings.body)
    assert est.height_px == pytest.approx(r.params.height_px, rel=0.02)
    start, stop = est.window
    assert start == 0 and stop < r.phases.bfc  # the window is the run-up, before the bound ends
    assert all(start <= f < stop for f in est.frames_used)


def test_height_scales_linearly(settings: Settings) -> None:
    a = estimate_height_px(generate(SynthParams(height_px=200)).sequence, settings.body)
    b = estimate_height_px(generate(SynthParams(height_px=400)).sequence, settings.body)
    assert b.height_px / a.height_px == pytest.approx(2.0, rel=1e-6)


def test_height_needs_the_head(settings: Settings) -> None:
    with pytest.raises(PoseQualityError, match="head"):
        estimate_height_px(_seq_without("nose"), settings.body)


def test_clip_without_runup_measures_height_from_segments(settings: Settings) -> None:
    """Shorts and reels start at the delivery stride: height comes from body segments."""
    seq = generate(SynthParams(fps=60, height_px=300.0)).sequence
    late = seq.model_copy(update={"frames": seq.frames[150:]})
    assert not has_runup(late, settings.body)
    assert runup_window(late, settings.body) == (0, len(late.frames))
    est = estimate_height_px(late, settings.body)
    assert est.method == "segments" and abs(est.height_px - 300.0) / 300.0 < 0.15


def test_mirror_twice_is_identity() -> None:
    seq = generate(SynthParams(noise_px=1.0, dropout_rate=0.05, seed=9)).sequence
    twice = mirror_sequence(mirror_sequence(seq))
    assert twice.bowling_arm == seq.bowling_arm
    for name in ("r_wrist", "l_ankle", "nose"):
        np.testing.assert_allclose(
            keypoint_track(twice, name), keypoint_track(seq, name), atol=1e-9, equal_nan=True
        )
    for a, b in zip(twice.frames, seq.frames, strict=True):
        assert {k for k, v in a.keypoints.items() if v is None} == {
            k for k, v in b.keypoints.items() if v is None
        }


def test_mirror_left_arm_equals_right_arm_flipped() -> None:
    right = generate(SynthParams()).sequence
    left = generate(SynthParams(bowling_arm="left")).sequence
    mirrored = mirror_sequence(left)
    assert mirrored.bowling_arm == "right"
    for name in ("r_wrist", "l_ankle", "nose", "r_knee"):
        np.testing.assert_allclose(
            keypoint_track(mirrored, name)[:, 0],
            left.width - keypoint_track(right, name)[:, 0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            keypoint_track(mirrored, name)[:, 1], keypoint_track(right, name)[:, 1], atol=1e-9
        )


def test_mirror_handles_missing_and_z() -> None:
    frame = FramePose(
        frame_idx=0,
        t=0.0,
        keypoints={"l_wrist": Keypoint(x=10, y=5, z=3.0, conf=0.8), "r_wrist": None},
    )
    seq = PoseSequence(fps=30, width=100, height=50, bowling_arm="left", frames=[frame])
    out = mirror_sequence(seq).frames[0]
    kp = out.get("r_wrist")
    assert kp is not None and (kp.x, kp.y, kp.z, kp.conf) == (90, 5, -3.0, 0.8)
    assert out.get("l_wrist") is None


def test_height_falls_back_to_whole_clip_when_runup_is_hidden() -> None:
    """If the full body never shows during the run-up, height comes from later frames."""
    s = load_settings()
    seq = generate(SynthParams(fps=120.0, height_px=400.0)).sequence
    full = estimate_height_px(seq, s.body)
    start, stop = full.window
    frames = [
        f.model_copy(update={"keypoints": {**f.keypoints, "nose": None}}) if start <= i < stop
        else f
        for i, f in enumerate(seq.frames)
    ]  # fmt: skip
    fallback = estimate_height_px(seq.model_copy(update={"frames": frames}), s.body)
    assert fallback.window == (0, len(seq.frames))
    assert fallback.height_px == pytest.approx(full.height_px, rel=0.25)


def _track(tid: int, n: int, height: float, step: float) -> PersonTrack:
    """A track of ``n`` frames, boxes ``height`` tall, moving ``step`` px a frame."""
    boxes = [(10 + i * step, 100.0, 60 + i * step, 100.0 + height) for i in range(n)]
    return PersonTrack(track_id=tid, frames=list(range(n)), boxes=boxes, confs=[0.9] * n)


def test_the_person_who_moves_most_is_the_bowler() -> None:
    bowler = _track(1, 300, 600, 8.0)  # long, big, runs a long way
    blip = _track(2, 15, 280, 1.0)
    walker = _track(3, 200, 550, 1.5)
    pick = select_bowler([blip, walker, bowler], 1280, 10, dominant_ratio=2.0)
    assert pick.track_id == 1


def test_tiny_distant_people_are_never_the_bowler() -> None:
    fielder = _track(1, 120, 70, 6.0)  # far away, a tenth of the bowler's size
    bowler = _track(2, 120, 600, 5.0)
    pick = select_bowler([fielder, bowler], 1280, 10, min_size_frac=0.4)
    assert pick.track_id == 2
