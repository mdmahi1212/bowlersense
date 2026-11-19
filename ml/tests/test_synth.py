"""Tests for the synthetic bowling skeleton generator (Phase 2)."""

from __future__ import annotations

import itertools
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from bowlsense.schemas import KEYPOINT_NAMES, PoseSequence
from synth.skeleton_gen import (
    HAND,
    SynthParams,
    SynthResult,
    generate,
    render_to_video,
)

_LIMBS: tuple[tuple[str, str], ...] = tuple(
    (f"{s}_{a}", f"{s}_{b}")
    for s in ("l", "r")
    for a, b in (
        ("shoulder", "elbow"),
        ("elbow", "wrist"),
        ("hip", "knee"),
        ("knee", "ankle"),
        ("ankle", "heel"),
        ("ankle", "toe"),
    )
)
SEGMENTS: tuple[tuple[str, str], ...] = (
    *_LIMBS,
    ("l_shoulder", "r_shoulder"),
    ("l_hip", "r_hip"),
    ("l_ear", "r_ear"),
)


@cache
def _gen(**kw: object) -> SynthResult:
    return generate(SynthParams.model_validate(kw))


def _xyz(seq: PoseSequence, name: str) -> np.ndarray:
    out = []
    for f in seq.frames:
        kp = f.get(name)
        assert kp is not None and kp.z is not None
        out.append((kp.x, kp.y, kp.z))
    return np.array(out)


def _xy(seq: PoseSequence, name: str) -> np.ndarray:
    return _xyz(seq, name)[:, :2]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1, 1))), dtype=np.float64)


def test_default_sequence_shape() -> None:
    r = _gen()
    seq = r.sequence
    assert seq.fps == 120 and seq.view == "side" and seq.bowling_arm == "right"
    assert len(seq.frames) == 462
    for f in seq.frames:
        assert set(f.keypoints) == set(KEYPOINT_NAMES)
        for kp in f.keypoints.values():
            assert kp is not None
            assert 0 <= kp.x < seq.width and 0 <= kp.y < seq.height
    assert r.phases.ordering_violations() == []
    assert r.phases.confidence == 1.0


@pytest.mark.parametrize(
    ("view", "arm"), list(itertools.product(("side", "front", "behind"), ("right", "left")))
)
def test_segment_lengths_constant(view: str, arm: str) -> None:
    seq = _gen(view=view, bowling_arm=arm).sequence
    for a, b in SEGMENTS:
        lengths = np.linalg.norm(_xyz(seq, a) - _xyz(seq, b), axis=1)
        assert lengths.std() / lengths.mean() < 1e-6, (a, b)


def test_segment_lengths_scale_with_height() -> None:
    short = _gen(height_px=150).sequence
    tall = _gen(height_px=300).sequence
    for a, b in SEGMENTS[:4]:
        ratio = np.linalg.norm(_xyz(tall, a)[0] - _xyz(tall, b)[0]) / np.linalg.norm(
            _xyz(short, a)[0] - _xyz(short, b)[0]
        )
        assert ratio == pytest.approx(2.0)


def test_left_arm_is_mirror_of_right() -> None:
    right = _gen().sequence
    left = _gen(bowling_arm="left").sequence
    for name in KEYPOINT_NAMES:
        mirrored = name.replace("l_", "#").replace("r_", "l_").replace("#", "r_")
        r_xyz, l_xyz = _xyz(right, name), _xyz(left, mirrored)
        np.testing.assert_allclose(l_xyz[:, :2], r_xyz[:, :2], atol=1e-9)
        np.testing.assert_allclose(l_xyz[:, 2], -r_xyz[:, 2], atol=1e-9)
    assert _gen(bowling_arm="left").phases == _gen().phases


@pytest.mark.parametrize("arm", ["right", "left"])
def test_bowling_wrist_is_fastest(arm: str) -> None:
    seq = _gen(bowling_arm=arm).sequence
    peak = {
        side: np.linalg.norm(np.diff(_xy(seq, f"{side}_wrist"), axis=0), axis=1).max()
        for side in ("l", "r")
    }
    bowling = "r" if arm == "right" else "l"
    other = "l" if bowling == "r" else "r"
    assert peak[bowling] > 1.5 * peak[other]


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"elbow_extension_deg": 22, "counter_rotation_deg": 35, "knee_flex_change_deg": 30},
        {"stride_pct": 95, "release_height_pct": 110, "runup_decel_pct": 25},
        {"stride_pct": 62, "release_height_pct": 125, "follow_through_pct": 30},
        {"bowling_arm": "left", "lateral_flexion_deg": 35, "fps": 240},
    ],
)
def test_truth_matches_params(kw: dict[str, object]) -> None:
    r = _gen(**kw)
    p, t = r.params, r.truth
    assert t.elbow_extension_deg == pytest.approx(p.elbow_extension_deg, abs=0.5)
    assert t.counter_rotation_deg == pytest.approx(p.counter_rotation_deg, abs=2.0)
    assert t.knee_flex_change_deg == pytest.approx(p.knee_flex_change_deg, abs=0.5)
    assert t.stride_pct == pytest.approx(p.stride_pct, abs=1.0)
    assert t.release_height_pct == pytest.approx(p.release_height_pct, abs=1.0)
    assert t.runup_decel_pct == pytest.approx(p.runup_decel_pct, abs=1.0)
    assert t.follow_through_pct == pytest.approx(p.follow_through_pct, abs=2.0)
    assert t.lateral_flexion_deg == pytest.approx(p.lateral_flexion_deg, abs=1.0)


@pytest.mark.parametrize(("extension", "arm"), [(8.0, "right"), (22.0, "left")])
def test_elbow_extension_visible_in_2d_side_view(extension: float, arm: str) -> None:
    """The arm moves in the camera plane, so the 2D elbow angle shows the full extension."""
    r = _gen(elbow_extension_deg=extension, bowling_arm=arm)
    seq, ph = r.sequence, r.phases
    side = "r" if arm == "right" else "l"
    angle = _angle(
        _xy(seq, f"{side}_shoulder"), _xy(seq, f"{side}_elbow"), _xy(seq, f"{side}_wrist")
    )
    window = angle[ph.arm_horizontal : ph.release + 1]
    k = int(np.argmax(window))
    assert window[k] - window[: k + 1].min() == pytest.approx(extension, abs=0.5)


def test_release_height_from_keypoints() -> None:
    r = _gen(release_height_pct=122)
    seq, s = r.sequence, r.params.height_px
    ground_y = max(kp.y for f in seq.frames for kp in f.keypoints.values() if kp is not None)
    ground_y += 0.004 * s  # toe tip sits 0.004 H above the ground
    wrist_y = _xy(seq, "r_wrist")[r.phases.release, 1]
    assert ((ground_y - wrist_y) / s + HAND) * 100 == pytest.approx(122, abs=1.0)


@pytest.mark.parametrize(
    ("action", "mixed"),
    [("side_on", False), ("semi_open", False), ("front_on", False), ("mixed", True)],
)
def test_action_types(action: str, mixed: bool) -> None:
    sep = abs(_gen(action_type=action).truth.hip_shoulder_sep_bfc_deg)
    assert (sep > 30) is mixed


def test_phase_times_independent_of_fps() -> None:
    a, b = _gen(fps=120).phases, _gen(fps=240).phases
    for name in ("bfc", "ffc", "arm_horizontal", "release"):
        assert getattr(b, name) / 240 == pytest.approx(getattr(a, name) / 120, abs=1 / 120)


def test_phase_events_sensible() -> None:
    r = _gen()
    seq, ph = r.sequence, r.phases
    hips = (_xy(seq, "l_hip") + _xy(seq, "r_hip")) / 2
    # bound = highest hip point (smallest image y) between run-up start and bfc
    assert ph.bound == ph.run_up_start + int(np.argmin(hips[ph.run_up_start : ph.bfc + 1, 1]))
    # feet are on the ground (largest image y) at their contacts
    floor_y = _xy(seq, "l_ankle")[:, 1].max()
    assert _xy(seq, "r_ankle")[ph.bfc, 1] == pytest.approx(floor_y, abs=1e-6)
    assert _xy(seq, "l_ankle")[ph.ffc, 1] == pytest.approx(floor_y, abs=1e-6)
    # the bowling wrist is above the shoulder at release
    assert _xy(seq, "r_wrist")[ph.release, 1] < _xy(seq, "r_shoulder")[ph.release, 1]


def test_views_differ_and_front_view_is_narrow() -> None:
    side, front = _gen(view="side").sequence, _gen(view="front").sequence
    assert side.width > 2 * front.width
    assert front.view == "front"
    assert _gen(view="behind").sequence.view == "behind"


def test_noise_and_dropout_statistics() -> None:
    clean = _gen().sequence
    noisy = _gen(noise_px=2.0, dropout_rate=0.1, seed=7).sequence
    diffs, missing, total = [], 0, 0
    for fc, fn in zip(clean.frames, noisy.frames, strict=True):
        for name in KEYPOINT_NAMES:
            total += 1
            kn, kc = fn.get(name), fc.get(name)
            if kn is None:
                missing += 1
                continue
            assert kc is not None
            diffs.append(kn.x - kc.x)
            assert 0.5 <= kn.conf <= 1.0
    assert missing / total == pytest.approx(0.1, abs=0.01)
    assert float(np.std(diffs)) == pytest.approx(2.0, rel=0.05)


def test_seed_determinism() -> None:
    a = generate(SynthParams(noise_px=1.0, dropout_rate=0.05, seed=3))
    b = generate(SynthParams(noise_px=1.0, dropout_rate=0.05, seed=3))
    c = generate(SynthParams(noise_px=1.0, dropout_rate=0.05, seed=4))
    assert a.sequence == b.sequence
    assert a.sequence != c.sequence


def test_include_z_false() -> None:
    seq = _gen(include_z=False).sequence
    assert all(kp is None or kp.z is None for kp in seq.frames[0].keypoints.values())


def test_unreachable_parameters_fail_loudly() -> None:
    with pytest.raises(ValueError, match="release_height_pct"):
        generate(SynthParams(release_height_pct=135))
    with pytest.raises(ValueError, match="stride_pct"):
        generate(SynthParams(stride_pct=105, runup_decel_pct=50))
    with pytest.raises(ValidationError):
        SynthParams(stride_pct=150)
    with pytest.raises(ValidationError):
        SynthParams(dropout_rate=1.0)


def test_low_fps_still_generates() -> None:
    r = generate(SynthParams(fps=30))
    assert r.phases.ordering_violations() == []


def test_result_json_round_trip() -> None:
    r = _gen()
    again = SynthResult.model_validate_json(r.model_dump_json())
    assert again == r


def test_render_to_video(tmp_path: Path) -> None:
    r = generate(SynthParams(fps=60, height_px=100))
    out = render_to_video(r.sequence, tmp_path / "sub" / "synth.mp4")
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(r.sequence.frames)
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == r.sequence.width
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == r.sequence.height
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(60)
        ok, frame = cap.read()
        assert ok and frame.max() > 150  # the bright skeleton was drawn
    finally:
        cap.release()


def test_render_to_unwritable_path_raises(tmp_path: Path) -> None:
    seq = generate(SynthParams(fps=30, height_px=60)).sequence
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        render_to_video(seq, blocker / "out.mp4")
