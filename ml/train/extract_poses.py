"""Extract the bowler's pose around every labelled delivery and cache it as JSON.

Uses the dataset's own bowler boxes (gaps of a few frames are interpolated), so detection
errors don't leak into the training data. Videos are decoded sequentially (not by seeking),
so every frame index matches the annotations exactly.

Run: ml/.venv/Scripts/python -m train.extract_poses --dataset C:/dev/dataset
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel

from bowlsense.config import REPO_ROOT, load_settings
from bowlsense.io.video import iter_frames, probe_video
from bowlsense.pose.base import load_backend
from bowlsense.schemas import KEYPOINT_NAMES, FramePose, PoseSequence
from train.dataset import Box, Delivery, bowler_boxes, dataset_dir, deliveries

log = logging.getLogger("train.extract")
DEFAULT_OUT = REPO_ROOT / "data" / "training" / "poses"
BEFORE_S, AFTER_S, MAX_BOX_GAP = 0.5, 1.5, 10


class ExtractedDelivery(BaseModel):
    """A delivery's label plus the bowler's pose over its window."""

    delivery: Delivery
    window_start: int  # video frame of sequence frame 0
    boxes_found: int
    sequence: PoseSequence

    @property
    def release_idx(self) -> int:
        """Release frame within the sequence."""
        return self.delivery.release - self.window_start

    @property
    def span_start_idx(self) -> int:
        """First "is bowling" frame within the sequence."""
        return self.delivery.span_start - self.window_start


def fill_boxes(boxes: dict[int, Box], first: int, last: int, max_gap: int) -> dict[int, Box]:
    """Linearly interpolate missing boxes across gaps of at most ``max_gap`` frames."""
    known = sorted(boxes)
    out = dict(boxes)
    for a, b in pairwise(known):
        if 1 < b - a <= max_gap + 1:
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                out[f] = tuple(  # type: ignore[assignment]
                    boxes[a][i] + w * (boxes[b][i] - boxes[a][i]) for i in range(4)
                )
    return {f: v for f, v in out.items() if first <= f <= last}


def main(argv: list[str] | None = None) -> int:
    """Extract every delivery that isn't cached yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", default="rtm", choices=["rtm", "mediapipe"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = dataset_dir(args.dataset)
    settings = load_settings()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    todo: dict[str, list[Delivery]] = defaultdict(list)
    for d in deliveries(root):
        if not (out / f"{d.key}.json").is_file():
            todo[d.video].append(d)
    if not todo:
        log.info("all deliveries already extracted in %s", out)
        return 0
    backend = load_backend(args.backend, settings.pose, settings.storage.models_path)
    total = sum(len(v) for v in todo.values())
    done, t0 = 0, time.time()
    for video, items in sorted(todo.items()):
        path = root / "videos" / f"{video}.mp4"
        info = probe_video(path)
        before, after = round(BEFORE_S * info.fps), round(AFTER_S * info.fps)
        windows: list[tuple[Delivery, int, int, dict[int, Box], list[FramePose]]] = []
        for d in items:
            first = max(0, d.span_start - before)
            last = min(info.frame_count - 1, d.release + after)
            boxes = fill_boxes(bowler_boxes(root, video, first, last), first, last, MAX_BOX_GAP)
            windows.append((d, first, last, boxes, []))
        stop = max(w[2] for w in windows) + 1
        for idx, frame in iter_frames(path, 0, stop):
            for _d, first, last, boxes, frames in windows:
                if first <= idx <= last:
                    box = boxes.get(idx)
                    kps = (
                        backend.estimate(frame, box)
                        if box is not None
                        else dict.fromkeys(KEYPOINT_NAMES)
                    )
                    k = idx - first
                    frames.append(FramePose(frame_idx=k, t=k / info.fps, keypoints=kps))
        for d, first, _last, boxes, frames in windows:
            seq = PoseSequence(fps=info.fps, width=info.width, height=info.height, frames=frames)
            item = ExtractedDelivery(
                delivery=d, window_start=first, boxes_found=len(boxes), sequence=seq
            )
            (out / f"{d.key}.json").write_text(item.model_dump_json(), encoding="utf-8")
        done += len(items)
        log.info("%s: %d deliveries (%d/%d, %.0f s)", video, len(items), done, total,
                 time.time() - t0)  # fmt: skip
    return 0


def load_extracted(folder: Path = DEFAULT_OUT) -> list[ExtractedDelivery]:
    """Every cached delivery, sorted by key."""
    return [
        ExtractedDelivery.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(folder.glob("*.json"))
    ]


if __name__ == "__main__":
    raise SystemExit(main())
