"""Optional 3D lifting (MotionBERT), behind the ``pose.lift_3d`` feature flag.

The system works fully without it: every module measures in 2D. When the flag is on, a
MotionBERT model exported to ONNX (``models/<pose.lift_model>``; see
``scripts/export_motionbert.py``) lifts each frame's 2D keypoints to 3D and fills
``Keypoint.z`` (same units as x/y: pixels, camera depth, relative to the hip centre).

MotionBERT takes Human3.6M-ordered joints, ``(1, T, 17, 3)`` = (x, y, confidence) with x/y
scaled to [-1, 1] by the image size, over windows of ``lift_window`` frames, and returns
``(1, T, 17, 3)`` in the same scaled units.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.errors import ModelNotFoundError
from bowlsense.schemas import Keypoint, PoseSequence

log = logging.getLogger(__name__)

# Human3.6M joint order used by MotionBERT, built from BowlSense names (pairs = midpoint).
H36M: tuple[tuple[str, ...], ...] = (
    ("l_hip", "r_hip"),  # 0 root (pelvis)
    ("r_hip",), ("r_knee",), ("r_ankle",),  # 1-3
    ("l_hip",), ("l_knee",), ("l_ankle",),  # 4-6
    ("__spine",),  # 7 midpoint of root and thorax
    ("l_shoulder", "r_shoulder"),  # 8 thorax
    ("nose",),  # 9 neck/nose
    ("l_eye", "r_eye"),  # 10 head
    ("l_shoulder",), ("l_elbow",), ("l_wrist",),  # 11-13
    ("r_shoulder",), ("r_elbow",), ("r_wrist",),  # 14-16
)  # fmt: skip
# BowlSense joint -> H36M index whose depth it takes.
DEPTH_FROM: dict[str, int] = {
    "nose": 9, "l_eye": 10, "r_eye": 10, "l_ear": 10, "r_ear": 10,
    "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13, "r_shoulder": 14, "r_elbow": 15,
    "r_wrist": 16, "l_hip": 4, "l_knee": 5, "l_ankle": 6, "l_heel": 6, "l_toe": 6,
    "r_hip": 1, "r_knee": 2, "r_ankle": 3, "r_heel": 3, "r_toe": 3,
}  # fmt: skip


def to_h36m(seq: PoseSequence) -> np.ndarray:
    """(T, 17, 3) MotionBERT input: x, y scaled to [-1, 1] by the image size, and confidence.

    Missing joints get confidence 0 (and the previous frame's position, or the centre).
    """
    t = len(seq.frames)
    out = np.zeros((t, 17, 3), dtype=np.float32)
    scale = min(seq.width, seq.height) / 2.0
    cx, cy = seq.width / 2.0, seq.height / 2.0
    for i, f in enumerate(seq.frames):
        for j, names in enumerate(H36M):
            if names == ("__spine",):
                continue
            pts = [f.keypoints.get(n) for n in names]
            if any(p is None for p in pts):
                out[i, j] = (out[i - 1, j, 0], out[i - 1, j, 1], 0.0) if i else (0.0, 0.0, 0.0)
                continue
            ok = [p for p in pts if p is not None]
            x = sum(p.x for p in ok) / len(ok)
            y = sum(p.y for p in ok) / len(ok)
            out[i, j] = ((x - cx) / scale, (y - cy) / scale, min(p.conf for p in ok))
        out[i, 7, :2] = (out[i, 0, :2] + out[i, 8, :2]) / 2.0
        out[i, 7, 2] = min(out[i, 0, 2], out[i, 8, 2])
    return out


def windows(t: int, size: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` windows covering ``t`` frames."""
    return [(s, min(t, s + size)) for s in range(0, t, size)] if t else []


class MotionBertLifter:
    """Runs the ONNX-exported MotionBERT on windows of frames."""

    def __init__(self, model: Path, window: int) -> None:
        """Load the ONNX model once."""
        import onnxruntime as ort

        from bowlsense.device import gpu_allowed

        providers = ["CPUExecutionProvider"]
        if gpu_allowed() and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session: Any = ort.InferenceSession(str(model), providers=providers)
        self.input = self.session.get_inputs()[0].name
        self.window = window

    def lift(self, x: np.ndarray) -> np.ndarray:
        """(T, 17, 3) scaled 2D + conf -> (T, 17, 3) scaled 3D. Short windows are edge-padded."""
        out = np.zeros_like(x)
        for start, stop in windows(len(x), self.window):
            chunk = x[start:stop]
            pad = self.window - len(chunk)
            if pad:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
            y = self.session.run(None, {self.input: chunk[None].astype(np.float32)})[0][0]
            out[start:stop] = y[: stop - start]
        return out


@lru_cache(maxsize=2)
def load_lifter(model: Path, window: int) -> MotionBertLifter:
    """Load (and cache) the lifter.

    Raises:
        ModelNotFoundError: if the ONNX file isn't in ``models/``.
    """
    if not model.is_file():
        raise ModelNotFoundError(
            model.name,
            model.parent,
            "3D lifting is optional (pose.lift_3d). Export MotionBERT with "
            "`python scripts/export_motionbert.py`, or set pose.lift_3d: false.",
        )
    return MotionBertLifter(model, window)


def lift_sequence(seq: PoseSequence, lifter: MotionBertLifter) -> PoseSequence:
    """``seq`` with ``z`` filled for every visible keypoint (pixels, relative to the hips)."""
    if not seq.frames:
        return seq
    lifted = lifter.lift(to_h36m(seq))
    scale = min(seq.width, seq.height) / 2.0
    frames = []
    for i, f in enumerate(seq.frames):
        root_z = float(lifted[i, 0, 2])
        kps: dict[str, Keypoint | None] = {}
        for name, k in f.keypoints.items():
            if k is None:
                kps[name] = None
                continue
            z = (float(lifted[i, DEPTH_FROM[name], 2]) - root_z) * scale
            kps[name] = k.model_copy(update={"z": z})
        frames.append(f.model_copy(update={"keypoints": kps}))
    log.info("3D lift: filled z for %d frames", len(frames))
    return seq.model_copy(update={"frames": frames})
