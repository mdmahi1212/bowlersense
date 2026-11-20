"""Auto-detected crease no-ball screen and the rough (uncalibrated) speed estimate."""

from __future__ import annotations

import cv2
import numpy as np

from bowlsense.analysis.noball import detect_no_ball, rough_speed_kmh
from bowlsense.config import load_settings
from bowlsense.schemas import FramePose, Keypoint

H = 300.0  # body height in pixels


def _pose(heel_x: float, back_x: float, y: float = 330.0) -> FramePose:
    kp = {
        "l_heel": Keypoint(x=heel_x, y=y, z=None, conf=0.9),  # front foot of a right-armer
        "l_ankle": Keypoint(x=heel_x, y=y - 8, z=None, conf=0.9),
        "r_ankle": Keypoint(x=back_x, y=y - 4, z=None, conf=0.9),
    }
    return FramePose(frame_idx=0, t=0.0, keypoints=kp)


def _pitch(with_crease: bool = True) -> np.ndarray:
    img = np.full((420, 480, 3), (40, 110, 60), np.uint8)  # grass green (BGR)
    if with_crease:
        cv2.line(img, (230, 400), (262, 260), (250, 250, 250), 4)  # steep white crease
    return img


def test_heel_behind_the_crease_is_legal() -> None:
    s = load_settings().delivery
    r = detect_no_ball(_pitch(), _pose(heel_x=215, back_x=120), "right", H, s)
    assert r.status == "legal" and r.margin_cm is not None and r.margin_cm < 0 and r.line


def test_heel_beyond_the_crease_is_a_no_ball() -> None:
    s = load_settings().delivery
    r = detect_no_ball(_pitch(), _pose(heel_x=275, back_x=150), "right", H, s)
    assert r.status == "no_ball" and r.margin_cm is not None and r.margin_cm > 0


def test_running_the_other_way_flips_the_side() -> None:
    s = load_settings().delivery
    img = np.asarray(cv2.flip(_pitch(), 1), dtype=np.uint8)  # mirror the scene
    r = detect_no_ball(img, _pose(heel_x=480 - 275, back_x=480 - 150), "right", H, s)
    assert r.status == "no_ball"


def test_no_crease_found() -> None:
    s = load_settings().delivery
    r = detect_no_ball(_pitch(False), _pose(heel_x=215, back_x=120), "right", H, s)
    assert r.status == "not_found" and r.margin_cm is None
    assert detect_no_ball(_pitch(), None, "right", H, s).status == "not_found"


def test_rough_speed_from_ball_travel() -> None:
    s = load_settings().delivery
    fps, release = 120.0, 10
    # 130 km/h with a 1.75 m bowler who is 300 px tall: px/s = v / 1.75 * 300
    px_s = (130 / 3.6) / s.assumed_height_m * H
    pts = [(release + i, 100 + px_s * i / fps, 200.0, 0.9) for i in range(1, 12)]
    v = rough_speed_kmh(pts, release, fps, H, "side", s)
    assert v is not None and abs(v - 130) < 2
    assert rough_speed_kmh(pts, release, fps, H, "front", s) is None  # depth: not measurable
    assert rough_speed_kmh(pts[:2], release, fps, H, "side", s) is None
    slow = [(f, 100 + 0.01 * (f - release), y, c) for f, _, y, c in pts]
    assert rough_speed_kmh(slow, release, fps, H, "side", s) is None  # implausible: tracking error
