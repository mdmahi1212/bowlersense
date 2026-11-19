"""MediaPipe Pose Landmarker (Tasks API) fallback backend -> BowlSense names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.detect.person import BBox, model_path
from bowlsense.io.video import Frame
from bowlsense.pose.base import KeypointMap, clamp_conf, pad_bbox
from bowlsense.schemas import KEYPOINT_NAMES, Keypoint

# MediaPipe's 33 landmarks -> BowlSense names (left/right = the person's own side).
MEDIAPIPE_TO_BOWLSENSE: dict[int, str] = {
    0: "nose",
    2: "l_eye",
    5: "r_eye",
    7: "l_ear",
    8: "r_ear",
    11: "l_shoulder",
    12: "r_shoulder",
    13: "l_elbow",
    14: "r_elbow",
    15: "l_wrist",
    16: "r_wrist",
    23: "l_hip",
    24: "r_hip",
    25: "l_knee",
    26: "r_knee",
    27: "l_ankle",
    28: "r_ankle",
    29: "l_heel",
    30: "r_heel",
    31: "l_toe",  # foot index
    32: "r_toe",
}


def map_mediapipe(landmarks: list[tuple[float, float, float]], crop: BBox) -> KeypointMap:
    """Turn normalised (x, y, visibility) landmarks of a crop into full-frame keypoints."""
    if len(landmarks) != 33:
        raise ValueError(f"expected 33 MediaPipe landmarks, got {len(landmarks)}")
    x1, y1, x2, y2 = crop
    cw, ch = x2 - x1, y2 - y1
    out: KeypointMap = dict.fromkeys(KEYPOINT_NAMES)
    for idx, name in MEDIAPIPE_TO_BOWLSENSE.items():
        nx, ny, vis = landmarks[idx]
        if np.isfinite(nx) and np.isfinite(ny):
            out[name] = Keypoint(x=x1 + nx * cw, y=y1 + ny * ch, z=None, conf=clamp_conf(vis))
    return out


class MediaPipeBackend:
    """MediaPipe Pose Landmarker (full) run on the bowler's padded crop."""

    name = "mediapipe"

    def __init__(self, weights: Path, padding: float) -> None:
        """Load the landmarker once."""
        from mediapipe.tasks.python import BaseOptions, vision

        self.padding = padding
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(weights)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker: Any = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame: Frame, bbox: BBox) -> KeypointMap:
        """Keypoints for the person in ``bbox`` (full-frame coordinates)."""
        import cv2
        import mediapipe as mp

        h, w = frame.shape[:2]
        crop = pad_bbox(bbox, self.padding, w, h)
        x1, y1, x2, y2 = (round(v) for v in crop)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return dict.fromkeys(KEYPOINT_NAMES)
        rgb = cv2.cvtColor(np.ascontiguousarray(frame[y1:y2, x1:x2]), cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return dict.fromkeys(KEYPOINT_NAMES)
        marks = [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]
        return map_mediapipe(marks, (float(x1), float(y1), float(x2), float(y2)))


def load_mediapipe(models_dir: Path, filename: str, padding: float) -> MediaPipeBackend:
    """Load the MediaPipe backend from ``models/``."""
    return MediaPipeBackend(model_path(models_dir, filename), padding)
