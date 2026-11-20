"""Reader for the bowling-release dataset (videos + per-frame JSON annotations + split CSV).

Layout::

    <dataset>/data_split.csv              video_name, camera_angle, n release events, set_name
    <dataset>/videos/<video_name>.mp4     30 fps match footage
    <dataset>/data/<video_name>.json      {"event": {frame: [...]}, "scene": {...}, "person": {...}}

Each delivery is an "is bowling" span that ends one frame before a single "bowl release"
frame. The release frame carries bowling type (fast/spin), bowler hand and over/around.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Split = Literal["train", "test", "challenge"]
Box = tuple[float, float, float, float]


class Delivery(BaseModel):
    """One labelled delivery."""

    model_config = ConfigDict(frozen=True)

    video: str
    split: Split
    span_start: int  # first "is bowling" frame
    release: int  # "bowl release" frame
    bowling_type: Literal["fast", "spin"] | None
    bowler_hand: Literal["right", "left"] | None
    bowling_from: str | None
    camera_angle: str
    bowl_direction: str | None  # at release: towards camera / away from camera / ...

    @property
    def key(self) -> str:
        """Stable id: ``<video>__<release frame>``."""
        return f"{self.video}__{self.release:06d}"


def dataset_dir(path: str | Path | None = None) -> Path:
    """The dataset folder from ``path`` or ``$BOWLSENSE_DATASET``.

    Raises:
        FileNotFoundError: if neither is given or the folder lacks ``data_split.csv``.
    """
    chosen = path or os.environ.get("BOWLSENSE_DATASET")
    if not chosen:
        raise FileNotFoundError("Pass --dataset or set BOWLSENSE_DATASET to the dataset folder.")
    root = Path(chosen)
    if not (root / "data_split.csv").is_file():
        raise FileNotFoundError(f"{root} has no data_split.csv; is it the dataset folder?")
    return root


def read_split(root: Path) -> dict[str, tuple[Split, str]]:
    """video name -> (split, camera angle)."""
    with (root / "data_split.csv").open(encoding="utf-8") as f:
        return {
            r["video_name"]: (r["set_name"], r["camera_angle"])  # type: ignore[misc]
            for r in csv.DictReader(f)
        }


def load_annotations(root: Path, video: str) -> dict[str, dict[str, list[dict[str, object]]]]:
    """The raw annotation JSON of one video."""
    data: dict[str, dict[str, list[dict[str, object]]]] = json.loads(
        (root / "data" / f"{video}.json").read_text(encoding="utf-8")
    )
    return data


def _norm(value: object, mapping: dict[str, str]) -> str | None:
    return mapping.get(str(value))


def deliveries(root: Path, splits: tuple[Split, ...] = ("train", "test")) -> list[Delivery]:
    """Every labelled delivery in the chosen splits, in video/time order."""
    out: list[Delivery] = []
    for video, (split, angle) in sorted(read_split(root).items()):
        if split not in splits or not (root / "data" / f"{video}.json").is_file():
            continue
        ann = load_annotations(root, video)
        events = {int(k): v[0] for k, v in ann["event"].items()}
        frames = sorted(events)
        for f in frames:
            e = events[f]
            if e["event"] != "bowl release":
                continue
            start = f - 1
            while start - 1 in events and events[start - 1]["event"] == "is bowling":
                start -= 1
            scene = ann["scene"].get(str(f), [{}])[0]
            out.append(
                Delivery(
                    video=video,
                    split=split,
                    span_start=start,
                    release=f,
                    bowling_type=_norm(e["bowling_type"], {"fast bowler": "fast",
                                                           "spin bowler": "spin"}),
                    bowler_hand=_norm(e["bowler_hand"], {"right-handed": "right",
                                                         "left-handed": "left"}),
                    bowling_from=_norm(e["bowling_from"], {"over the wicket": "over",
                                                           "around the wicket": "around"}),
                    camera_angle=angle,
                    bowl_direction=str(scene.get("bowl_direction")) if scene else None,
                )
            )  # fmt: skip
    return out


def bowler_boxes(root: Path, video: str, first: int, last: int) -> dict[int, Box]:
    """The annotated bowler box per frame in ``[first, last]`` (frames without one are
    skipped)."""
    people = load_annotations(root, video)["person"]
    out: dict[int, Box] = {}
    for f in range(first, last + 1):
        for p in people.get(str(f), []):
            if p["role"] == "bowler":
                b = p["box"]
                assert isinstance(b, list)
                out[f] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                break
    return out
