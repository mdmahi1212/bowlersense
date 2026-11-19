"""Pose backend interface and the shared crop helper."""

from __future__ import annotations

from typing import Protocol

from bowlsense.detect.person import BBox
from bowlsense.io.video import Frame
from bowlsense.schemas import Keypoint

KeypointMap = dict[str, Keypoint | None]


class PoseBackend(Protocol):
    """Estimates the 21 BowlSense keypoints of one person in a frame."""

    name: str

    def estimate(self, frame: Frame, bbox: BBox) -> KeypointMap:
        """Keypoints in full-frame pixel coordinates (``None`` = not detected)."""
        ...


def pad_bbox(bbox: BBox, padding: float, width: int, height: int) -> BBox:
    """Grow a box by ``padding`` of its size on each side, clipped to the frame."""
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0.0, x1 - pw),
        max(0.0, y1 - ph),
        min(float(width), x2 + pw),
        min(float(height), y2 + ph),
    )


def clamp_conf(value: float) -> float:
    """Model scores aren't always in [0, 1]; keypoint confidence must be."""
    return float(min(1.0, max(0.0, value)))


def load_backend(name: str, settings: object, models_dir: object) -> PoseBackend:
    """Create a pose backend by name ("rtm" or "mediapipe") from ``models/``.

    ``settings`` is ``PoseSettings``; ``models_dir`` a ``Path``. Models are loaded once per
    call; callers keep the backend for the whole video.
    """
    from pathlib import Path

    from bowlsense.config import PoseSettings
    from bowlsense.errors import ConfigError

    if not isinstance(settings, PoseSettings):
        raise TypeError("settings must be PoseSettings")
    folder = Path(str(models_dir))
    if name == "rtm":
        from bowlsense.pose.rtm import load_rtm

        return load_rtm(folder, settings.rtm_model, settings.rtm_input_size, settings.crop_padding)
    if name == "mediapipe":
        from bowlsense.pose.mediapipe_backend import load_mediapipe

        return load_mediapipe(folder, settings.mediapipe_model, settings.crop_padding)
    raise ConfigError(f"Unknown pose backend '{name}'; use 'rtm' or 'mediapipe'.")
