"""Person detection + ByteTrack tracking (Ultralytics YOLO) and bowler selection."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.config import DetectSettings
from bowlsense.errors import ModelNotFoundError, NoBowlerDetectedError
from bowlsense.io.video import Frame, iter_frames

log = logging.getLogger(__name__)
PERSON_CLASS = 0
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class PersonTrack(BaseModel):
    """One tracked person: a bounding box per frame where they were seen."""

    model_config = ConfigDict(frozen=True)

    track_id: int
    frames: list[int]
    boxes: list[BBox]
    confs: list[float]

    def box_at(self, frame: int) -> BBox | None:
        """The box in ``frame``, or ``None`` if not seen then."""
        try:
            return self.boxes[self.frames.index(frame)]
        except ValueError:
            return None


def model_path(models_dir: Path, filename: str) -> Path:
    """Path to a weights file in ``models/``.

    Raises:
        ModelNotFoundError: if it hasn't been downloaded yet.
    """
    path = models_dir / filename
    if not path.is_file():
        raise ModelNotFoundError(filename, models_dir)
    return path


def _new_tracker_model(weights: Path) -> Any:
    """A fresh YOLO model (tracker state is per instance, so one per video)."""
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})  # no telemetry
    return YOLO(str(weights))


def _device() -> str:
    from bowlsense.device import gpu_allowed

    if not gpu_allowed():
        log.info("detector: GPU turned off in settings, using CPU")
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            log.info("detector: using CUDA GPU")
            return "cuda:0"
    except ImportError:  # pragma: no cover - torch is a dependency of ultralytics
        pass
    log.info("detector: no CUDA GPU found, using CPU")
    return "cpu"


def track_people(
    frames: Iterable[tuple[int, Frame]], weights: Path, settings: DetectSettings
) -> list[PersonTrack]:
    """Run YOLO person detection with ByteTrack over ``frames``; return the tracks."""
    model = _new_tracker_model(weights)
    data: dict[int, tuple[list[int], list[BBox], list[float]]] = {}
    device = _device()
    for idx, frame in frames:
        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=settings.person_conf,
            tracker=settings.tracker,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for tid, box, conf in zip(ids, xyxy, confs, strict=True):
            fr, bx, cf = data.setdefault(int(tid), ([], [], []))
            fr.append(idx)
            bx.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
            cf.append(float(conf))
    return [
        PersonTrack(track_id=tid, frames=fr, boxes=bx, confs=cf)
        for tid, (fr, bx, cf) in sorted(data.items())
    ]


def track_people_in_video(
    video: Path, models_dir: Path, settings: DetectSettings
) -> list[PersonTrack]:
    """``track_people`` over every frame of a video file."""
    return track_people(iter_frames(video), model_path(models_dir, settings.yolo_model), settings)


def track_motion(track: PersonTrack, frame_width: int) -> tuple[float, float]:
    """(horizontal displacement, robust peak speed) of a track's box centre, both as a
    fraction of frame width (speed per frame). The 95th percentile is used as the peak so
    a single ID-switch jump can't dominate."""
    if len(track.frames) < 2:
        return 0.0, 0.0
    cx = np.array([(b[0] + b[2]) / 2.0 for b in track.boxes])
    frames = np.array(track.frames, dtype=np.float64)
    displacement = float(cx.max() - cx.min()) / frame_width
    steps = np.abs(np.diff(cx)) / np.maximum(np.diff(frames), 1.0)
    peak = float(np.percentile(steps, 95)) / frame_width
    return displacement, peak


def track_features(track: PersonTrack) -> tuple[float, float]:
    """(2D extent, size change) of a track, both camera-distance independent.

    Extent: how far the box centre travels (5th-95th percentile range in x and y) in units
    of the person's median box height. Size change: log of the 95th / 5th percentile box
    height, large when someone runs towards or away from the camera.
    """
    if len(track.frames) < 2:
        return 0.0, 0.0
    b = np.asarray(track.boxes, dtype=np.float64)
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
    span = np.hypot(np.ptp(np.percentile(cx, [5, 95])), np.ptp(np.percentile(cy, [5, 95])))
    extent = float(span / np.median(h))
    size = float(np.log(np.percentile(h, 95) / np.percentile(h, 5)))
    return extent, size


def select_bowler(
    tracks: list[PersonTrack],
    frame_width: int,
    min_frames: int,
    override_id: int | None = None,
) -> PersonTrack:
    """The bowler, or the ``override_id`` track if given.

    Each usable track is scored on three things, each normalised by the best track:
    how far it moves across the image (any direction, in body heights, so a side-on run-up
    and a run towards the camera both count), how much its size changes (approaching or
    leaving the camera) and how much of the clip it covers (the bowler is the subject).
    ``frame_width`` is kept for the API; the features are scale-free.

    Raises:
        NoBowlerDetectedError: if no usable track exists or ``override_id`` isn't one.
    """
    if override_id is not None:
        for t in tracks:
            if t.track_id == override_id:
                return t
        ids = ", ".join(str(t.track_id) for t in tracks) or "none"
        raise NoBowlerDetectedError(
            f"Track id {override_id} was not found (tracks in this clip: {ids})."
        )
    usable = [t for t in tracks if len(t.frames) >= min_frames]
    if not usable:
        raise NoBowlerDetectedError(
            f"No person was tracked for at least {min_frames} frames "
            f"({len(tracks)} short track(s) found)."
        )
    feats = [track_features(t) for t in usable]
    longest = max(len(t.frames) for t in usable)
    max_extent = max(f[0] for f in feats) or 1.0
    max_size = max(f[1] for f in feats) or 1.0
    scores = [
        e / max_extent + sz / max_size + len(t.frames) / longest
        for (e, sz), t in zip(feats, usable, strict=True)
    ]
    best = int(np.argmax(scores))
    log.info(
        "bowler: track %d (score %.2f of %d tracks)",
        usable[best].track_id,
        scores[best],
        len(usable),
    )
    return usable[best]
