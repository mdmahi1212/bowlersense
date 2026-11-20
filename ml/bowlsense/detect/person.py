"""Person detection + ByteTrack tracking (Ultralytics YOLO) and bowler selection."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.config import DetectSettings
from bowlsense.errors import ModelNotFoundError, NoBowlerDetectedError
from bowlsense.geometry.angles import FloatArray
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


class TrackScorer(Protocol):
    """Anything that scores tracks as "bowler" (e.g. ``detect.selector.BowlerSelector``)."""

    def scores(
        self, tracks: list[PersonTrack], width: int, height: int, n_frames: int
    ) -> FloatArray: ...


def select_bowler(
    tracks: list[PersonTrack],
    frame_width: int,
    min_frames: int,
    override_id: int | None = None,
    *,
    scorer: TrackScorer | None = None,
    frame_height: int = 0,
    n_frames: int = 0,
) -> PersonTrack:
    """The bowler, or the ``override_id`` track if given.

    With a ``scorer`` (the selector trained on labelled match footage) the top-scoring track
    is chosen. Otherwise the rule: largest horizontal displacement plus highest peak
    horizontal speed, each normalised by the best among the tracks. On 279 labelled match
    deliveries the rule picked the bowler 70.6% of the time (the tracker had the bowler
    93.9% of the time); see ``data/training/reports``.

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
    if scorer is not None:
        height = frame_height or frame_width
        span = n_frames or max(t.frames[-1] for t in usable) + 1
        scores = [float(v) for v in scorer.scores(usable, frame_width, height, span)]
        method = "learned"
    else:
        motions = [track_motion(t, frame_width) for t in usable]
        max_disp = max(m[0] for m in motions) or 1.0
        max_speed = max(m[1] for m in motions) or 1.0
        scores = [d / max_disp + sp / max_speed for d, sp in motions]
        method = "rule"
    best = int(np.argmax(scores))
    log.info(
        "bowler: track %d (%s score %.2f of %d tracks)",
        usable[best].track_id,
        method,
        scores[best],
        len(usable),
    )
    return usable[best]
