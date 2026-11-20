"""High-accuracy pose options: flip test-time augmentation, optical-flow wrist refinement,
3D elbow angles from lifted depth, and the accurate/fast model profile."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bowlsense.analysis.legality import arm_is_3d, elbow_angles
from bowlsense.config import Settings, load_settings
from bowlsense.local_settings import accuracy_profile
from bowlsense.pose.refine import refine_wrist
from bowlsense.pose.rtm import HALPE26_FLIP, flip_average
from bowlsense.schemas import FramePose, Keypoint, PoseSequence


def test_flip_average_unmirrors_and_swaps_sides() -> None:
    rng = np.random.default_rng(0)
    width = 640
    kp = rng.uniform(50, 500, size=(26, 2))
    sc = rng.uniform(0.5, 1.0, size=26)
    mirrored = kp[HALPE26_FLIP].copy()  # what the model sees on the flipped image
    mirrored[:, 0] = (width - 1) - mirrored[:, 0]
    merged, conf = flip_average(kp, sc, mirrored, sc[HALPE26_FLIP], width)
    assert np.allclose(merged, kp) and np.allclose(conf, sc)
    shifted = mirrored.copy()
    shifted[:, 0] -= 10  # the flipped run is 10 px off -> the average moves half way
    merged, _ = flip_average(kp, np.ones(26), shifted, np.ones(26), width)
    assert np.allclose(merged[:, 0], kp[:, 0] + 5)


def _dot_video(path: Path, centres: list[tuple[float, float]]) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 100.0, (320, 240))
    try:
        for x, y in centres:
            img = np.zeros((240, 320, 3), np.uint8)
            cv2.rectangle(img, (40, 40), (280, 200), (60, 60, 60), -1)  # some texture
            cv2.circle(img, (round(x), round(y)), 6, (255, 255, 255), -1)
            cv2.circle(img, (round(x) + 3, round(y) - 2), 2, (0, 0, 0), -1)
            w.write(img)
    finally:
        w.release()


def test_wrist_refinement_fills_gaps_and_rejects_jumps(tmp_path: Path) -> None:
    truth = [(80.0 + 4 * i, 120.0 - 2 * i) for i in range(30)]
    video = tmp_path / "dot.mp4"
    _dot_video(video, truth)
    frames = []
    for i, (x, y) in enumerate(truth):
        wrist: Keypoint | None = Keypoint(x=x, y=y, z=None, conf=0.9)
        if i in (14, 15):
            wrist = None  # dropped out
        if i == 17:
            wrist = Keypoint(x=x + 60, y=y + 50, z=None, conf=0.9)  # jumped (blur)
        nose = Keypoint(x=160, y=20, z=None, conf=0.9)
        ankle = Keypoint(x=160, y=220, z=None, conf=0.9)
        frames.append(FramePose(frame_idx=i, t=i / 100, keypoints={"r_wrist": wrist, "nose": nose,
                                                                   "l_ankle": ankle}))  # fmt: skip
    seq = PoseSequence(fps=100, width=320, height=240, view="side", bowling_arm="right",
                       frames=frames)  # fmt: skip
    s = load_settings().pose
    out = refine_wrist(video, seq, [15], s)
    for i in (14, 15, 17):
        k = out.frames[i].keypoints["r_wrist"]
        assert k is not None, i
        assert np.hypot(k.x - truth[i][0], k.y - truth[i][1]) < 3.0, (i, k)
    assert out.frames[5].keypoints["r_wrist"] == seq.frames[5].keypoints["r_wrist"]  # untouched


def _arm_seq(z_wrist: float | None) -> PoseSequence:
    def kp(x: float, y: float, z: float | None) -> Keypoint:
        return Keypoint(x=x, y=y, z=z, conf=0.9)

    frames = [
        FramePose(
            frame_idx=i,
            t=i / 100,
            keypoints={
                "r_shoulder": kp(100, 100, None if z_wrist is None else 0.0),
                "r_elbow": kp(100, 150, None if z_wrist is None else 0.0),
                "r_wrist": kp(100, 200, z_wrist),  # straight in 2D; bent in 3D when z_wrist set
            },
        )
        for i in range(5)
    ]
    return PoseSequence(fps=100, width=400, height=400, view="side", bowling_arm="right",
                        frames=frames)  # fmt: skip


def test_elbow_uses_3d_depth_when_lifted() -> None:
    flat = _arm_seq(None)
    assert not arm_is_3d(flat) and np.allclose(elbow_angles(flat), 180.0)
    bent = _arm_seq(-50.0)  # forearm swings 50 px towards the camera: a 135 deg elbow
    assert arm_is_3d(bent) and np.allclose(elbow_angles(bent), 135.0)


@pytest.fixture()
def models_dir_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = load_settings()
    for name in (s.pose.accurate_rtm_model, s.detect.accurate_yolo_model, s.pose.lift_model):
        (tmp_path / name).write_bytes(b"x")
    storage = s.storage.model_copy(update={"models_dir": tmp_path})
    monkeypatch.delenv("BOWLSENSE_ACCURACY", raising=False)
    return s.model_copy(update={"storage": storage})


def test_accuracy_profile_swaps_in_the_big_models(models_dir_settings: Settings) -> None:
    s = models_dir_settings
    acc = accuracy_profile(s.model_copy(update={"pose": s.pose.model_copy(
        update={"accuracy": "accurate"})}))  # fmt: skip
    assert acc.pose.rtm_model == s.pose.accurate_rtm_model
    assert acc.pose.rtm_input_size == s.pose.accurate_rtm_input_size
    assert acc.pose.use_flip_test and acc.pose.lift_3d
    assert acc.detect.yolo_model == s.detect.accurate_yolo_model
    assert acc.ball.imgsz == s.ball.accurate_imgsz
    fast = accuracy_profile(s.model_copy(update={"pose": s.pose.model_copy(
        update={"accuracy": "fast"})}))  # fmt: skip
    assert fast.pose.rtm_model == s.pose.rtm_model and not fast.pose.use_flip_test


def test_missing_accurate_files_keep_the_fast_models(tmp_path: Path) -> None:
    s = load_settings()
    s = s.model_copy(update={
        "storage": s.storage.model_copy(update={"models_dir": tmp_path}),
        "pose": s.pose.model_copy(update={"accuracy": "accurate"}),
    })  # fmt: skip
    acc = accuracy_profile(s)
    assert acc.pose.rtm_model == s.pose.rtm_model and acc.detect.yolo_model == s.detect.yolo_model
