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
    colours: list[list[float]] = []  # per frame: torso colour histogram (see torso_colours)

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


HIST_BINS = (12, 4)  # hue x saturation


def torso_colours(frame: Frame, box: BBox) -> list[float]:
    """Normalised hue-saturation histogram of the torso (middle of the box): clothing colour,
    which tells a bowler from a batter or umpire when the tracker mixes them up."""
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + 0.25 * w)), int(min(frame.shape[1], x2 - 0.25 * w))
    ya, yb = int(max(0, y1 + 0.2 * h)), int(min(frame.shape[0], y1 + 0.6 * h))
    if xb - xa < 2 or yb - ya < 2:
        return [0.0] * (HIST_BINS[0] * HIST_BINS[1])
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256]).ravel()
    total = float(hist.sum())
    return [float(v) / total for v in hist] if total > 0 else [0.0] * hist.size


def colour_distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Hellinger distance between two histograms (0 = same colours, 1 = no overlap)."""
    pa, pb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(pa * pb))))))


def track_people(
    frames: Iterable[tuple[int, Frame]], weights: Path, settings: DetectSettings
) -> list[PersonTrack]:
    """Run YOLO person detection with ByteTrack over ``frames``; return the tracks (with the
    torso colours of every box)."""
    model = _new_tracker_model(weights)
    data: dict[int, tuple[list[int], list[BBox], list[float], list[list[float]]]] = {}
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
            fr, bx, cf, co = data.setdefault(int(tid), ([], [], [], []))
            b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            fr.append(idx)
            bx.append(b)
            cf.append(float(conf))
            co.append(torso_colours(frame, b))
    return [
        PersonTrack(track_id=tid, frames=fr, boxes=bx, confs=cf, colours=co)
        for tid, (fr, bx, cf, co) in sorted(data.items())
    ]


def track_people_in_video(
    video: Path, models_dir: Path, settings: DetectSettings, fps: float = 30.0
) -> list[PersonTrack]:
    """``track_people`` over every frame of a video file, with broken tracks stitched."""
    tracks = track_people(iter_frames(video), model_path(models_dir, settings.yolo_model), settings)
    return stitch_for(tracks, settings, fps)


def stitch_for(
    tracks: list[PersonTrack], settings: DetectSettings, fps: float
) -> list[PersonTrack]:
    """Split identity swaps by clothing colour, then stitch broken tracks (configured limits;
    times in seconds of file time)."""
    window = max(3, round(settings.colour_window_s * fps))
    splits: set[tuple[int, int]] = set()
    pieces = split_on_colour(
        tracks,
        window,
        settings.colour_split_threshold,
        min_piece=settings.min_track_frames,
        splits=splits,
    )
    return stitch_tracks(
        pieces,
        max_gap_frames=round(settings.stitch_max_gap_s * fps),
        max_jump=settings.stitch_max_jump,
        max_size_ratio=settings.stitch_max_size_ratio,
        max_colour_distance=settings.colour_join_threshold,
        forbidden=splits,
    )


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


def _piece(t: PersonTrack, lo: int, hi: int, track_id: int) -> PersonTrack:
    return PersonTrack(
        track_id=track_id, frames=t.frames[lo:hi], boxes=t.boxes[lo:hi], confs=t.confs[lo:hi],
        colours=t.colours[lo:hi] if t.colours else [],
    )  # fmt: skip


def split_on_colour(
    tracks: list[PersonTrack],
    window: int,
    threshold: float,
    min_piece: int,
    splits: set[tuple[int, int]] | None = None,
) -> list[PersonTrack]:
    """Split tracks whose clothing colours belong to two different people: the tracker
    swapped identities (e.g. the bowler running past the batter or umpire).

    For every split point the mean colours of *all* frames before and *all* frames after are
    compared (Hellinger distance), so a gradual swap during an overlap is found and a brief
    glitch is not. The best point splits the track when the distance exceeds ``threshold``
    and both pieces are at least ``max(window, min_piece)`` frames; pieces are split again.
    New pieces get new ids; each (before, after) id pair is added to ``splits`` so stitching
    won't join them back together.
    """
    next_id = max((t.track_id for t in tracks), default=0) + 1
    shortest = max(window, min_piece)
    out: list[PersonTrack] = []
    todo = list(tracks)
    while todo:
        t = todo.pop()
        n = len(t.frames)
        if not t.colours or n < 2 * shortest:
            out.append(t)
            continue
        c = np.asarray(t.colours, dtype=np.float64)
        csum = np.vstack([np.zeros((1, c.shape[1])), np.cumsum(c, axis=0)])
        best_i, best_d = -1, threshold
        for i in range(shortest, n - shortest + 1):
            d = colour_distance(csum[i] / i, (csum[n] - csum[i]) / (n - i))
            if d > best_d:
                best_i, best_d = i, d
        if best_i < 0:
            out.append(t)
            continue
        log.info("track %d: another person's colours from frame %d (%.2f); split", t.track_id,
                 t.frames[best_i], best_d)  # fmt: skip
        todo += [_piece(t, 0, best_i, t.track_id), _piece(t, best_i, n, next_id)]
        if splits is not None:
            splits.add((t.track_id, next_id))
        next_id += 1
    return sorted(out, key=lambda t: t.track_id)


def _mean_colours(colours: list[list[float]]) -> np.ndarray | None:
    return np.mean(np.asarray(colours, dtype=np.float64), axis=0) if colours else None


def stitch_tracks(
    tracks: list[PersonTrack],
    max_gap_frames: int,
    max_jump: float = 0.5,
    max_size_ratio: float = 1.5,
    max_colour_distance: float | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[PersonTrack]:
    """Join tracks that are the same person after the tracker lost them for a moment.

    A track B continues track A when B starts after A ends (no overlap) within
    ``max_gap_frames``, B's first box centre is within ``max_jump`` box heights of A's last
    one, and their heights differ by less than ``max_size_ratio``. Each track continues at
    most one other; the closest candidate wins. The joined track keeps A's id.
    """
    ordered = sorted(tracks, key=lambda t: t.frames[0])
    successor: dict[int, int] = {}
    taken: set[int] = set()
    pairs = []
    for a in ordered:
        ax1, ay1, ax2, ay2 = a.boxes[-1]
        ah = max(ay2 - ay1, 1.0)
        for b in ordered:
            gap = b.frames[0] - a.frames[-1]
            if b is a or gap <= 0 or gap > max_gap_frames:
                continue
            if forbidden and (a.track_id, b.track_id) in forbidden:
                continue
            bx1, by1, bx2, by2 = b.boxes[0]
            bh = max(by2 - by1, 1.0)
            if max(ah, bh) / min(ah, bh) > max_size_ratio:
                continue
            jump = np.hypot((bx1 + bx2 - ax1 - ax2) / 2.0, (by1 + by2 - ay1 - ay2) / 2.0)
            if max_colour_distance is not None:
                # whole-track colours: at the join itself the boxes often overlap another person
                ca = _mean_colours(a.colours)
                cb = _mean_colours(b.colours)
                if (
                    ca is not None
                    and cb is not None
                    and (colour_distance(ca, cb) > max_colour_distance)
                ):
                    continue
            if jump <= max_jump * max(ah, bh):
                pairs.append((jump / max(ah, bh), gap, a.track_id, b.track_id))
    for _, _, a_id, b_id in sorted(pairs):
        if a_id not in successor and b_id not in taken:
            successor[a_id], taken = b_id, taken | {b_id}
    by_id = {t.track_id: t for t in tracks}
    out = []
    for t in ordered:
        if t.track_id in taken:
            continue  # joined onto its predecessor
        frames, boxes, confs = list(t.frames), list(t.boxes), list(t.confs)
        colours = list(t.colours)
        nxt = successor.get(t.track_id)
        while nxt is not None:
            n = by_id[nxt]
            frames += n.frames
            boxes += n.boxes
            confs += n.confs
            colours = colours + n.colours if colours and n.colours else []
            nxt = successor.get(nxt)
        out.append(
            PersonTrack(
                track_id=t.track_id, frames=frames, boxes=boxes, confs=confs, colours=colours
            )
        )
    return out


class TrackScorer(Protocol):
    """Anything that scores tracks as "bowler" (e.g. ``detect.selector.BowlerSelector``)."""

    def scores(
        self, tracks: list[PersonTrack], width: int, height: int, n_frames: int
    ) -> FloatArray: ...


def track_travel(track: PersonTrack) -> float:
    """Total distance (pixels) the box centre moves over the track."""
    if len(track.boxes) < 2:
        return 0.0
    c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in track.boxes])
    return float(np.abs(np.diff(c, axis=0)).sum())


def select_bowler(
    tracks: list[PersonTrack],
    frame_width: int,
    min_frames: int,
    override_id: int | None = None,
    *,
    scorer: TrackScorer | None = None,
    frame_height: int = 0,
    n_frames: int = 0,
    min_size_frac: float = 0.0,
    dominant_ratio: float = 0.0,
) -> PersonTrack:
    """The bowler, or the ``override_id`` track if given.

    With a ``scorer`` (the selector trained on labelled match footage) the top-scoring track
    is chosen. Otherwise the rule: largest horizontal displacement plus highest peak
    horizontal speed, each normalised by the best among the tracks. On 279 labelled match
    deliveries the rule picked the bowler 70.6% of the time (the tracker had the bowler
    93.9% of the time); see ``data/training/reports``.

    ``min_size_frac`` drops tracks whose median box height is below that fraction of the
    tallest track's (distant fielders and spectators) before scoring. ``dominant_ratio``:
    a track whose total travel is that many times anyone else's is the bowler outright.

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
    if min_size_frac > 0 and len(usable) > 1:
        # a bowler is never tiny next to the biggest person in the clip: drop distant
        # fielders and spectators before scoring (they fooled the selector on phone clips)
        sizes = [float(np.median([b[3] - b[1] for b in t.boxes])) for t in usable]
        big = max(sizes)
        kept = [t for t, h in zip(usable, sizes, strict=True) if h >= min_size_frac * big]
        if kept:
            usable = kept
    if dominant_ratio > 0 and len(usable) > 1:
        # the bowler is the person who moves most: when one track's travel is at least
        # ``dominant_ratio`` x everyone else's, take it without asking the scorer
        moves = sorted(((track_travel(t), i) for i, t in enumerate(usable)), reverse=True)
        if moves[0][0] >= dominant_ratio * max(moves[1][0], 1.0):
            chosen = usable[moves[0][1]]
            log.info("bowler: track %d (moves %.0f px, %.1fx anyone else)", chosen.track_id,
                     moves[0][0], moves[0][0] / max(moves[1][0], 1.0))  # fmt: skip
            return chosen
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
