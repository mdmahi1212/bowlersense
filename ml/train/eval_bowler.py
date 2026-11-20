"""Real-footage benchmark for bowler selection (the step that picks which tracked person is
the bowler), against the dataset's per-frame "bowler" boxes.

For each labelled delivery the clip window the app analyses (release - ``clipping.before_s``
to release + ``clipping.after_s``) is tracked with YOLO + ByteTrack exactly as in the app.
Then:
  - selected: the app's ``select_bowler`` choice,
  - legacy:   the old horizontal-motion-only choice (for comparison),
  - oracle:   the best single track (upper bound set by the tracker itself).
A choice is correct when its box overlaps the labelled bowler (IoU >= 0.5) in at least half
of the labelled frames of the window.

Run: ml/.venv/Scripts/python -m train.eval_bowler [--split test|train|all] [--limit N]
Writes data/training/reports/bowler_selection.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import load_settings
from bowlsense.detect.person import (
    BBox,
    PersonTrack,
    model_path,
    select_bowler,
    track_motion,
    track_people,
)
from bowlsense.errors import NoBowlerDetectedError
from bowlsense.io.video import iter_frames, probe_video
from train.dataset import bowler_boxes, dataset_dir, deliveries
from train.train_release import REPORTS

log = logging.getLogger("train.eval_bowler")
IOU_MATCH = 0.5
TRACK_CACHE = REPORTS.parent / "tracks"  # data/training/tracks: cached tracks per delivery
MIN_MATCH_FRACTION = 0.5


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_fraction(track: PersonTrack, labels: dict[int, BBox]) -> float:
    """Share of labelled frames where the track's box matches the labelled bowler."""
    if not labels:
        return 0.0
    hits = 0
    for f, box in labels.items():
        mine = track.box_at(f)
        hits += mine is not None and iou(mine, box) >= IOU_MATCH
    return hits / len(labels)


def legacy_select(tracks: list[PersonTrack], width: int, min_frames: int) -> PersonTrack:
    """The previous rule: horizontal displacement + peak horizontal speed only."""
    usable = [t for t in tracks if len(t.frames) >= min_frames]
    if not usable:
        raise NoBowlerDetectedError("no usable track")
    motions = [track_motion(t, width) for t in usable]
    md = max(m[0] for m in motions) or 1.0
    ms = max(m[1] for m in motions) or 1.0
    return usable[int(np.argmax([d / md + s / ms for d, s in motions]))]


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark, print a summary and save the report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["test", "train", "all"], default="test")
    p.add_argument("--limit", type=int, default=0, help="only the first N deliveries")
    p.add_argument("--dataset", type=Path, help="dataset folder (or $BOWLSENSE_DATASET)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("bowlsense", "ultralytics"):
        logging.getLogger(name).setLevel(logging.WARNING)
    s = load_settings()
    root = dataset_dir(args.dataset)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    items = deliveries(root, splits)  # type: ignore[arg-type]
    if args.limit:
        items = items[: args.limit]
    weights = model_path(s.storage.models_path, s.detect.yolo_model)
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, d in enumerate(items):
        video = root / "videos" / f"{d.video}.mp4"
        info = probe_video(video)
        lo = max(0, d.release - round(s.clipping.before_release_s * info.fps))
        hi = min(info.frame_count, d.release + round(s.clipping.after_release_s * info.fps) + 1)
        labels = bowler_boxes(root, d.video, lo, hi - 1)
        if not labels:
            continue
        cache = TRACK_CACHE / f"{d.video}_{d.release}.json"
        if cache.is_file():
            tracks = [
                PersonTrack.model_validate(t) for t in json.loads(cache.read_text())["tracks"]
            ]
        else:
            tracks = track_people(iter_frames(video, lo, hi), weights, s.detect)
            TRACK_CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "video": d.video, "release": d.release, "split": d.split, "lo": lo, "hi": hi,
                "width": info.width, "height": info.height,
                "labels": {str(k): v for k, v in labels.items()},
                "tracks": [t.model_dump() for t in tracks],
            }), encoding="utf-8")  # fmt: skip
        row: dict[str, Any] = {
            "video": d.video, "release": d.release, "camera_angle": d.camera_angle,
            "direction": d.bowl_direction, "tracks": len(tracks),
        }  # fmt: skip
        row["oracle"] = max((match_fraction(t, labels) for t in tracks), default=0.0)
        pickers: dict[str, Callable[[list[PersonTrack], int, int], PersonTrack]] = {
            "selected": select_bowler,
            "legacy": legacy_select,
        }
        for name, pick in pickers.items():
            try:
                row[name] = match_fraction(
                    pick(tracks, info.width, s.detect.min_track_frames), labels
                )
            except NoBowlerDetectedError:
                row[name] = 0.0
        rows.append(row)
        if (i + 1) % 10 == 0:
            log.info("%d/%d deliveries (%.0f s)", i + 1, len(items), time.monotonic() - t0)

    def rate(key: str, subset: list[dict[str, Any]]) -> float | None:
        return sum(r[key] >= MIN_MATCH_FRACTION for r in subset) / len(subset) if subset else None

    by_dir: dict[str, dict[str, Any]] = {}
    for direction in sorted({str(r["direction"]) for r in rows}):
        sub = [r for r in rows if str(r["direction"]) == direction]
        by_dir[direction] = {k: rate(k, sub) for k in ("selected", "legacy", "oracle")}
        by_dir[direction]["n"] = len(sub)
    failures = [
        r for r in rows if r["selected"] < MIN_MATCH_FRACTION <= r["oracle"]
    ]  # selection errors (the tracker had the bowler)
    report = {
        "protocol": (
            f"split={args.split}; window = release -{s.clipping.before_release_s}s/"
            f"+{s.clipping.after_release_s}s; correct = IoU>={IOU_MATCH} in >= "
            f"{MIN_MATCH_FRACTION:.0%} of labelled frames"
        ),
        "n": len(rows),
        "selected_accuracy": rate("selected", rows),
        "legacy_accuracy": rate("legacy", rows),
        "oracle_accuracy": rate("oracle", rows),
        "by_bowl_direction": by_dir,
        "selection_failures": Counter(str(r["direction"]) for r in failures),
        "failed_deliveries": [
            {k: r[k] for k in ("video", "release", "direction", "selected", "oracle")}
            for r in failures
        ],
        "seconds": round(time.monotonic() - t0),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "bowler_selection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(json.dumps({k: v for k, v in report.items() if k != "failed_deliveries"}, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
