"""Video and image reading (frame by frame; a clip is never loaded fully into memory)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from bowlsense.errors import InvalidInputError, UnsupportedMediaError

Frame = NDArray[np.uint8]
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


class VideoInfo(BaseModel):
    """Basic properties of a video file."""

    model_config = ConfigDict(frozen=True)

    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        """Clip length in seconds."""
        return self.frame_count / self.fps


def is_image(path: Path | str) -> bool:
    """True for a supported still-image file extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _check_suffix(path: Path, allowed: tuple[str, ...]) -> None:
    suffix = path.suffix.lower()
    if suffix not in allowed:
        raise UnsupportedMediaError(suffix, allowed)


def _open(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise InvalidInputError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(
            f"Could not open {path.name} as a video. It may be corrupt or use a codec OpenCV "
            "can't read; re-export it as H.264 MP4."
        )
    return cap


def probe_video(path: Path | str) -> VideoInfo:
    """Read fps, size and frame count without decoding the whole clip.

    Raises:
        UnsupportedMediaError: if the extension isn't a supported video type.
        InvalidInputError: if the file is missing, unreadable, or reports no fps or frames.
    """
    p = Path(path)
    _check_suffix(p, VIDEO_EXTENSIONS)
    cap = _open(p)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise InvalidInputError(f"{p.name} does not report a frame rate; re-export the video.")
    if width <= 0 or height <= 0 or count <= 0:
        raise InvalidInputError(f"{p.name} has no readable frames.")
    return VideoInfo(path=str(p), fps=fps, width=width, height=height, frame_count=count)


def iter_frames(
    path: Path | str, start: int = 0, stop: int | None = None
) -> Iterator[tuple[int, Frame]]:
    """Yield ``(frame_index, BGR frame)`` from ``start`` up to (not including) ``stop``.

    The capture is always released, even if the caller stops early.
    """
    if start < 0:
        raise InvalidInputError("start must be >= 0")
    p = Path(path)
    _check_suffix(p, VIDEO_EXTENSIONS)
    cap = _open(p)
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while stop is None or idx < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, np.asarray(frame, dtype=np.uint8)
            idx += 1
    finally:
        cap.release()


def read_frame(path: Path | str, index: int) -> Frame:
    """Read one frame by index.

    Raises:
        InvalidInputError: if the frame doesn't exist.
    """
    for _, frame in iter_frames(path, index, index + 1):
        return frame
    raise InvalidInputError(f"Frame {index} is beyond the end of {Path(path).name}.")


def load_image(path: Path | str) -> Frame:
    """Load a still image as BGR.

    Raises:
        UnsupportedMediaError / InvalidInputError: for a bad extension or unreadable file.
    """
    p = Path(path)
    _check_suffix(p, IMAGE_EXTENSIONS)
    if not p.is_file():
        raise InvalidInputError(f"Image file not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)  # handles non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidInputError(f"Could not read {p.name} as an image.")
    return np.asarray(img, dtype=np.uint8)
