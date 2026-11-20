"""RTMPose (rtmlib, ONNX Runtime) backend, Halpe-26 keypoints -> BowlSense names."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.detect.person import BBox, model_path
from bowlsense.io.video import Frame
from bowlsense.pose.base import KeypointMap, clamp_conf, pad_bbox
from bowlsense.schemas import KEYPOINT_NAMES, Keypoint

log = logging.getLogger(__name__)

# Halpe-26 index -> BowlSense name (17 COCO joints + heels and big toes).
HALPE26_TO_BOWLSENSE: dict[int, str] = {
    0: "nose",
    1: "l_eye",
    2: "r_eye",
    3: "l_ear",
    4: "r_ear",
    5: "l_shoulder",
    6: "r_shoulder",
    7: "l_elbow",
    8: "r_elbow",
    9: "l_wrist",
    10: "r_wrist",
    11: "l_hip",
    12: "r_hip",
    13: "l_knee",
    14: "r_knee",
    15: "l_ankle",
    16: "r_ankle",
    20: "l_toe",  # big toe
    21: "r_toe",
    24: "l_heel",
    25: "r_heel",
}


def map_halpe26(keypoints: np.ndarray, scores: np.ndarray) -> KeypointMap:
    """Turn one person's Halpe-26 output ((26, 2) pixels, (26,) scores) into BowlSense
    keypoints."""
    if keypoints.shape != (26, 2) or scores.shape != (26,):
        raise ValueError(f"expected Halpe-26 output, got {keypoints.shape} / {scores.shape}")
    out: KeypointMap = dict.fromkeys(KEYPOINT_NAMES)
    for idx, name in HALPE26_TO_BOWLSENSE.items():
        x, y = float(keypoints[idx, 0]), float(keypoints[idx, 1])
        if np.isfinite(x) and np.isfinite(y):
            out[name] = Keypoint(x=x, y=y, z=None, conf=clamp_conf(float(scores[idx])))
    return out


def _ort_device() -> str:
    from bowlsense.device import gpu_allowed, onnx_cuda_ready

    if not gpu_allowed():
        log.info("pose: GPU turned off in settings, using CPU")
        return "cpu"
    if onnx_cuda_ready():
        log.info("pose: using CUDA GPU (onnxruntime)")
        return "cuda"
    log.info("pose: no usable CUDA for onnxruntime, using CPU")
    return "cpu"


class RtmBackend:
    """RTMPose-m Halpe-26 run on the bowler's padded box."""

    name = "rtm"

    def __init__(self, weights: Path, input_size: tuple[int, int], padding: float) -> None:
        """Load the ONNX model once."""
        from rtmlib import RTMPose

        self.padding = padding
        self._model: Any = RTMPose(
            str(weights), model_input_size=input_size, backend="onnxruntime", device=_ort_device()
        )

    def estimate(self, frame: Frame, bbox: BBox) -> KeypointMap:
        """Keypoints for the person in ``bbox`` (full-frame coordinates)."""
        h, w = frame.shape[:2]
        box = pad_bbox(bbox, self.padding, w, h)
        keypoints, scores = self._model(frame, [list(box)])
        return map_halpe26(np.asarray(keypoints)[0], np.asarray(scores)[0])


def load_rtm(
    models_dir: Path, filename: str, input_size: tuple[int, int], padding: float
) -> RtmBackend:
    """Load the RTMPose backend from ``models/``."""
    return RtmBackend(model_path(models_dir, filename), input_size, padding)
