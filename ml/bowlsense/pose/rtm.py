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


# Halpe-26 left/right pairs, swapped when un-mirroring a flipped image.
HALPE26_FLIP = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24]
)


def flip_average(
    kp: np.ndarray, sc: np.ndarray, kp_f: np.ndarray, sc_f: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a prediction with one made on the horizontally mirrored image: un-mirror the
    second (x -> width - 1 - x, left/right swapped) and average, weighted by confidence."""
    back = kp_f[HALPE26_FLIP].copy()
    back[:, 0] = (width - 1) - back[:, 0]
    sb = sc_f[HALPE26_FLIP]
    w1 = np.clip(sc, 1e-3, None)[:, None]
    w2 = np.clip(sb, 1e-3, None)[:, None]
    return (kp * w1 + back * w2) / (w1 + w2), (sc + sb) / 2.0


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
    """RTMPose (m or x) Halpe-26 run on the bowler's padded box."""

    name = "rtm"

    def __init__(
        self, weights: Path, input_size: tuple[int, int], padding: float, flip: bool = False
    ) -> None:
        """Load the ONNX model once. ``flip`` = flip test-time augmentation (also run on the
        mirrored crop and average): steadier joints and fewer left/right swaps, ~2x slower."""
        from rtmlib import RTMPose

        self.padding = padding
        self.flip = flip
        self._model: Any = RTMPose(
            str(weights), model_input_size=input_size, backend="onnxruntime", device=_ort_device()
        )

    def estimate(self, frame: Frame, bbox: BBox) -> KeypointMap:
        """Keypoints for the person in ``bbox`` (full-frame coordinates)."""
        h, w = frame.shape[:2]
        box = pad_bbox(bbox, self.padding, w, h)
        keypoints, scores = self._model(frame, [list(box)])
        kp, sc = np.asarray(keypoints, dtype=np.float64)[0], np.asarray(scores, dtype=np.float64)[0]
        if self.flip:
            mirrored = np.ascontiguousarray(frame[:, ::-1])
            x0, y0, x1, y1 = box
            kf, sf = self._model(mirrored, [[w - 1 - x1, y0, w - 1 - x0, y1]])
            kp, sc = flip_average(
                kp, sc, np.asarray(kf, dtype=np.float64)[0], np.asarray(sf, dtype=np.float64)[0], w
            )
        return map_halpe26(kp, sc)


def load_rtm(
    models_dir: Path,
    filename: str,
    input_size: tuple[int, int],
    padding: float,
    flip: bool = False,
) -> RtmBackend:
    """Load the RTMPose backend from ``models/``."""
    return RtmBackend(model_path(models_dir, filename), input_size, padding, flip)
