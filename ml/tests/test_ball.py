"""Tests for module F (Phase 9): calibration, ball tracking, speed, length, line."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from bowlsense.analysis.delivery import (
    BATTER_STUMPS_X,
    assess_ball,
    ball_speed_kmh,
    length_zone,
    line_band,
)
from bowlsense.config import Settings, load_settings
from bowlsense.detect.ball import BallKalman, Detector, find_bounce, track_ball
from bowlsense.errors import CalibrationError
from bowlsense.geometry.pitch import (
    CREASE_LENGTH_M,
    along_pitch,
    calibrate,
    image_to_pitch,
    is_side_on,
    pitch_corners,
    pitch_to_image,
)
from bowlsense.schemas import BallTrack
from synth.ball_gen import ANGLED, BEHIND, SIDE_ON, Camera, flight


@cache
def _settings() -> Settings:
    return load_settings()


def _detections(
    cam: Camera, speed: float, fps: float, noise: float, drop_every: int = 0, seed: int = 0
) -> tuple[dict[int, list[tuple[float, float, float]]], int]:
    """Per-frame candidates: the ball (with noise / drops) plus a static distractor."""
    world, bounce = flight(speed, fps)
    px = cam.project(world)
    rng = np.random.default_rng(seed)
    out: dict[int, list[tuple[float, float, float]]] = {}
    for i, (u, v) in enumerate(px):
        cands = [(800.0, 100.0, 0.9)]  # a sightscreen blob far from the ball's path
        if not (drop_every and i % drop_every == drop_every - 1 and i > 2):
            cands.append((u + rng.normal(0, noise), v + rng.normal(0, noise), 0.4))
        out[i] = cands
    return out, bounce


def _feeder(dets: dict[int, list[tuple[float, float, float]]], n: int) -> Detector:
    """A fake detector that returns the next frame's prepared candidates on each call."""
    frames = iter(range(n))
    return lambda _frame, _centre: dets.get(next(frames), [])


def _track(
    cam: Camera, speed: float, fps: float, noise: float = 0.5, drop_every: int = 0
) -> tuple[BallTrack, int]:
    dets, bounce = _detections(cam, speed, fps, noise, drop_every)
    hand = (dets[0][-1][0], dets[0][-1][1])
    frames = ((i, np.zeros((1, 1, 3), np.uint8)) for i in range(len(dets)))
    s = _settings()
    detector = _feeder(dets, len(dets))
    track = track_ball(frames, detector, 0, hand, fps, s.ball, s.delivery.first_ball_within_frames)
    return track, bounce


# --- gate -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("speed", [90.0, 120.0, 145.0])
@pytest.mark.parametrize("fps", [120.0, 240.0])
@pytest.mark.parametrize("cam", [SIDE_ON, ANGLED], ids=["side_on", "angled_11deg"])
def test_gate_speed_within_5pct(speed: float, fps: float, cam: Camera) -> None:
    """Phase 9 gate: a synthetic ball of known speed is recovered within +-5%."""
    track, _ = _track(cam, speed, fps, noise=0.5, drop_every=5)
    cal = calibrate(cam.crease_points().tolist())
    m = assess_ball(track, cal, fps, _settings().delivery)
    assert m.speed_kmh is not None
    assert m.speed_kmh == pytest.approx(speed, rel=0.05)


def test_gate_no_calibration_no_speed() -> None:
    """Phase 9 gate: without calibration, speed is None (never a guess)."""
    track, _ = _track(SIDE_ON, 130.0, 120.0)
    m = assess_ball(track, None, 120.0, _settings().delivery)
    assert m.speed_kmh is None and m.length is None and m.pitch_xy is None
    assert "calibration" in m.notes[0]


def test_behind_camera_reports_no_speed() -> None:
    track, _ = _track(BEHIND, 130.0, 120.0)
    cal = calibrate(BEHIND.crease_points().tolist())
    assert not is_side_on(cal)
    m = assess_ball(track, cal, 120.0, _settings().delivery)
    assert m.speed_kmh is None and any("side-on" in n for n in m.notes)


# --- bounce, length, line, pitch map --------------------------------------------------------------


@pytest.mark.parametrize("cam", [SIDE_ON, ANGLED, BEHIND], ids=["side", "angled", "behind"])
def test_bounce_and_pitch_position(cam: Camera) -> None:
    """Whatever is reported is accurate; what the view can't resolve is withheld with a
    note. Side-on cameras resolve length, cameras behind the bowler resolve line."""
    world, bounce = flight(120.0, 240.0)
    track, _ = _track(cam, 120.0, 240.0, noise=0.3)
    assert track.bounce_frame is not None and abs(track.bounce_frame - bounce) <= 2
    cal = calibrate(cam.crease_points().tolist())
    s = _settings().delivery
    m = assess_ball(track, cal, 240.0, s)
    assert m.pitch_xy is not None
    true_x, true_y = world[bounce, 0], world[bounce, 1]
    if m.length is not None:
        assert m.pitch_xy[0] == pytest.approx(true_x, abs=s.max_length_error_m)
        assert m.length == length_zone(BATTER_STUMPS_X - true_x, s)
    else:
        assert any("length not reported" in n for n in m.notes)
    if m.line is not None:
        assert m.pitch_xy[1] == pytest.approx(true_y, abs=s.max_line_error_m)
        assert m.line == line_band(true_y, s)
    else:
        assert any("line not reported" in n for n in m.notes)
    if cam is SIDE_ON:
        assert m.length is not None and m.line is None
    if cam is BEHIND:
        assert m.line is not None


@pytest.mark.parametrize(
    ("d", "zone"),
    [
        (0.5, "yorker"),
        (2.0, "full"),
        (5.9, "full"),
        (7.0, "good"),
        (9.0, "short"),
        (12.0, "bouncer"),
    ],
)
def test_length_zones(d: float, zone: str) -> None:
    assert length_zone(d, _settings().delivery) == zone


def test_line_bands_and_batter_hand() -> None:
    s = _settings().delivery
    assert line_band(0.0, s) == "middle"
    assert line_band(0.1, s) == "off stump" and line_band(-0.1, s) == "leg stump"
    assert line_band(0.5, s) == "outside off" and line_band(-0.5, s) == "outside leg"
    assert line_band(1.2, s) == "wide"
    lefty = s.model_copy(update={"batter_hand": "left"})
    assert line_band(0.5, lefty) == "outside leg"


# --- calibration ----------------------------------------------------------------------------------


def test_calibration_round_trip() -> None:
    cal = calibrate(SIDE_ON.crease_points().tolist())
    np.testing.assert_allclose(
        image_to_pitch(cal, SIDE_ON.crease_points()), pitch_corners(), atol=1e-4
    )
    pts = np.array([[3.0, 0.5], [12.0, -0.8]])
    np.testing.assert_allclose(image_to_pitch(cal, pitch_to_image(cal, pts)), pts, atol=1e-4)


def test_along_pitch_matches_centre_line() -> None:
    cal = calibrate(SIDE_ON.crease_points().tolist())
    xs = np.array([0.0, 5.0, CREASE_LENGTH_M])
    cols = SIDE_ON.project(np.stack([xs, np.zeros(3), np.zeros(3)], axis=1))[:, 0]
    np.testing.assert_allclose(along_pitch(cal, cols), xs, atol=0.02)
    assert np.isnan(along_pitch(cal, np.array([-5000.0]))).all()


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([[0, 0], [1, 1], [2, 2]], "exactly 4"),
        ([[0, 0], [0.5, 0.5], [100, 0], [100, 100]], "same"),
        ([[0, 0], [50, 0], [100, 0], [100, 100]], "one line"),
        ([[0, 0], [1, float("nan")], [100, 0], [100, 100]], "finite"),
    ],
)
def test_calibration_validation(points: list[list[float]], match: str) -> None:
    with pytest.raises(CalibrationError, match=match):
        calibrate(points)


# --- tracker --------------------------------------------------------------------------------------


def test_tracker_ignores_distractor_and_bridges_gaps() -> None:
    track, _ = _track(SIDE_ON, 130.0, 120.0, noise=0.5, drop_every=4)
    assert track.release_frame == 0
    assert all(abs(y - 100.0) > 50 for _, _, y, _ in track.points)  # never jumped to the blob
    bridged = [p for p in track.points if p[3] == 0.0]
    assert bridged and track.points[-1][3] > 0  # gaps bridged, never ends on a guess


def test_tracker_gives_up_after_max_missed() -> None:
    s = _settings()
    dets = {0: [(100.0, 100.0, 0.5)], 1: [(110.0, 101.0, 0.5)]}
    frames = ((i, np.zeros((1, 1, 3), np.uint8)) for i in range(40))
    track = track_ball(frames, _feeder(dets, 40), 0, (100.0, 100.0), 120.0, s.ball, 3)
    assert [p[0] for p in track.points] == [0, 1]


def test_tracker_needs_ball_near_hand() -> None:
    s = _settings()
    frames = ((i, np.zeros((1, 1, 3), np.uint8)) for i in range(10))
    track = track_ball(frames, lambda _f, _c: [(900.0, 600.0, 0.9)], 0, (100.0, 100.0), 120.0,
                       s.ball, 3)  # fmt: skip
    assert track.points == [] and track.release_frame is None
    m = assess_ball(track, calibrate(SIDE_ON.crease_points().tolist()), 120.0, _settings().delivery)
    assert m.speed_kmh is None and "not found" in m.notes[0]


def test_kalman_learns_gravity_free_motion() -> None:
    kf = BallKalman(0.0, 0.0, 100.0, _settings().ball)
    for i in range(1, 30):
        kf.predict()
        kf.update(np.array([10.0 * i, 2.0 * i]))
    assert kf.x[2] == pytest.approx(1000.0, rel=0.02) and kf.x[3] == pytest.approx(200.0, rel=0.05)


def test_find_bounce_and_speed_needs_points() -> None:
    ys = [10, 20, 30, 40, 50, 60, 55, 50, 45, 40, 35]  # down 5 frames, then up
    t = BallTrack(points=[(i, float(i), float(y), 1.0) for i, y in enumerate(ys)])
    assert find_bounce(t) == 5
    assert find_bounce(BallTrack(points=[(0, 0, 10, 1), (1, 1, 20, 1)])) is None
    cal = calibrate(SIDE_ON.crease_points().tolist())
    assert ball_speed_kmh(t, cal, 120.0, _settings().delivery) is None  # < 5 points
