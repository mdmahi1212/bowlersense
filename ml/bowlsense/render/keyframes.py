"""Keyframe stills (PNG) at the delivery's key events, with the overlay drawn."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bowlsense.config import RenderSettings
from bowlsense.errors import InvalidInputError
from bowlsense.io.video import Frame, read_frame
from bowlsense.render.overlay import EVENT_LABELS, OverlayPainter, OverlaySegment


def save_png(image: Frame, path: Path) -> Path:
    """Write a PNG (works with non-ASCII Windows paths)."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise InvalidInputError(f"Could not encode {path.name} as PNG.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(buf).tofile(str(path))
    return path


def export_keyframes(
    video: Path,
    painter: OverlayPainter,
    segment: OverlaySegment,
    out_dir: Path,
    settings: RenderSettings,
) -> dict[str, Path]:
    """One PNG per ``render.keyframe_events`` event of one delivery; event -> file path."""
    paths: dict[str, Path] = {}
    for event in settings.keyframe_events:
        idx = segment.start + int(getattr(segment.phases, event))
        image = painter.paint(idx, read_frame(video, idx))
        name = f"{list(EVENT_LABELS).index(event) + 1:02d}_{event}.png"
        paths[event] = save_png(image, out_dir / name)
    return paths
