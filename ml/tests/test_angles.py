"""Tests for geometry/angles.py using hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from bowlsense.errors import InvalidInputError
from bowlsense.geometry.angles import (
    angular_velocity,
    elevation_angle,
    joint_angle,
    keypoint_array,
    keypoint_track,
    line_tilt,
    linear_speed,
    midpoint,
    segment_angle,
    to_angle_series,
    valid_runs,
    velocity,
)
from bowlsense.schemas import FramePose, Keypoint, PoseSequence

NAN = float("nan")


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        ((1, 0), (0, 0), (0, 1), 90.0),
        ((1, 0), (0, 0), (-1, 0), 180.0),
        ((1, 0), (0, 0), (1, 0), 0.0),
        ((1, 0), (0, 0), (1, 1), 45.0),
        ((2, 0), (0, 0), (-1, np.sqrt(3)), 120.0),
        ((0, 0, 1), (0, 0, 0), (0, 1, 0), 90.0),
        ((3, 4, 0), (0, 0, 0), (-3, -4, 0), 180.0),
        # elbow: shoulder above, wrist forward -> right angle; scale-invariant
        ((100, 100), (100, 300), (400, 300), 90.0),
    ],
)
def test_joint_angle_hand_computed(
    a: tuple[float, ...], b: tuple[float, ...], c: tuple[float, ...], expected: float
) -> None:
    assert float(joint_angle(a, b, c)) == pytest.approx(expected, abs=1e-9)


def test_joint_angle_collinear_float_noise_is_not_nan() -> None:
    # 0.1+0.2 style rounding pushes cos just past -1; must clip, not NaN
    a = np.array([0.1 + 0.2, 0.0])
    angle = joint_angle(a, (0.0, 0.0), (-0.3, 0.0))
    assert float(angle) == pytest.approx(180.0)


def test_joint_angle_vectorised_and_nan() -> None:
    a = np.array([[1, 0], [NAN, 0], [1, 0], [0, 0]], dtype=float)
    b = np.zeros((4, 2))
    c = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=float)
    out = joint_angle(a, b, c)
    assert out[0] == pytest.approx(90)
    assert np.isnan(out[1])  # missing point
    assert np.isnan(out[3])  # zero-length segment


@pytest.mark.parametrize(
    ("end", "expected"),
    [((1, 0), 0.0), ((0, -1), 90.0), ((0, 1), -90.0), ((-1, 0), 180.0), ((1, -1), 45.0)],
)
def test_segment_angle_image_coords(end: tuple[float, float], expected: float) -> None:
    assert float(segment_angle((0, 0), end)) == pytest.approx(expected)


def test_segment_angle_zero_length_is_nan() -> None:
    assert np.isnan(segment_angle((5, 5), (5, 5)))


@pytest.mark.parametrize(
    ("end", "expected"), [((1, 0), 0.0), ((-1, 0), 0.0), ((-1, -1), 45.0), ((1, 1), -45.0)]
)
def test_elevation_ignores_left_right(end: tuple[float, float], expected: float) -> None:
    assert float(elevation_angle((0, 0), end)) == pytest.approx(expected)


def test_line_tilt_order_independent() -> None:
    assert float(line_tilt((0, 0), (10, -10))) == pytest.approx(45)
    assert float(line_tilt((10, -10), (0, 0))) == pytest.approx(45)
    assert float(line_tilt((0, 0), (10, 0))) == pytest.approx(0)


def test_angular_velocity_constant_rate() -> None:
    fps = 100.0
    angles = np.arange(0, 50) * 2.0  # 2 deg per frame -> 200 deg/s
    np.testing.assert_allclose(angular_velocity(angles, fps), 200.0)


def test_angular_velocity_unwraps_across_180() -> None:
    angles = np.array([170.0, 175.0, 180.0, -175.0, -170.0])
    np.testing.assert_allclose(angular_velocity(angles, 10.0), 50.0)


def test_angular_velocity_nan_runs() -> None:
    angles = np.array([0.0, 10.0, NAN, 50.0, 60.0, 70.0, NAN])
    out = angular_velocity(angles, 1.0)
    np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], 10.0)
    assert np.isnan(out[2]) and np.isnan(out[6])


def test_linear_speed() -> None:
    t = np.arange(20)
    pts = np.stack([3.0 * t, 4.0 * t], axis=1)  # 5 px per frame
    np.testing.assert_allclose(linear_speed(pts, 120.0), 600.0)
    pts[5] = NAN
    out = linear_speed(pts, 120.0)
    assert np.isnan(out[5]) and out[10] == pytest.approx(600.0)


def test_linear_speed_isolated_point_is_nan() -> None:
    pts = np.array([[0.0, 0.0], [NAN, NAN], [1.0, 1.0], [NAN, NAN]])
    assert np.all(np.isnan(linear_speed(pts, 30)))


def test_velocity_1d_and_2d() -> None:
    np.testing.assert_allclose(velocity(np.arange(5.0) * 2, 10), 20.0)
    v2 = velocity(np.stack([np.arange(5.0), np.zeros(5)], axis=1), 10)
    np.testing.assert_allclose(v2[:, 0], 10.0)


@pytest.mark.parametrize("fps", [0.0, -5.0, float("nan")])
def test_bad_fps_raises(fps: float) -> None:
    with pytest.raises(InvalidInputError):
        linear_speed(np.zeros((3, 2)), fps)
    with pytest.raises(InvalidInputError):
        angular_velocity(np.zeros(3), fps)


def test_linear_speed_shape_check() -> None:
    with pytest.raises(InvalidInputError):
        linear_speed(np.zeros(5), 30)


def test_valid_runs() -> None:
    assert valid_runs([True, True, False, True, False, False, True]) == [(0, 2), (3, 4), (6, 7)]
    assert valid_runs([False, False]) == []
    assert valid_runs([]) == []
    with pytest.raises(InvalidInputError):
        valid_runs(np.zeros((2, 2), dtype=bool))


def _seq() -> PoseSequence:
    frames = [
        FramePose(
            frame_idx=i,
            t=i / 30,
            keypoints={
                "nose": Keypoint(x=float(i), y=2.0, conf=0.9),
                "l_wrist": None if i == 1 else Keypoint(x=5.0, y=float(i), conf=0.5),
            },
        )
        for i in range(3)
    ]
    return PoseSequence(fps=30, width=100, height=100, frames=frames)


def test_keypoint_array_and_track() -> None:
    seq = _seq()
    xy, conf = keypoint_array(seq)
    assert xy.shape == (3, 21, 2) and conf.shape == (3, 21)
    track = keypoint_track(seq, "l_wrist")
    assert np.isnan(track[1]).all() and track[2].tolist() == [5.0, 2.0]
    assert keypoint_track(seq, "nose")[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(keypoint_track(seq, "r_toe")).all()
    with pytest.raises(InvalidInputError):
        keypoint_track(seq, "left_wrist")


def test_midpoint_and_angle_series() -> None:
    assert midpoint((0, 0), (2, 4)).tolist() == [1, 2]
    s = to_angle_series("elbow", [170.0, NAN, 175.0], frames=[10, 11, 12])
    assert s.values == [170.0, None, 175.0] and s.frames == [10, 11, 12]
    assert to_angle_series("x", [1.0, 2.0]).frames == [0, 1]
    with pytest.raises(InvalidInputError):
        to_angle_series("x", [1.0], frames=[0, 1])
