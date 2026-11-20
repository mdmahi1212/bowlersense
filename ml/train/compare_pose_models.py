"""Compare pose models on the labelled real-footage test videos.

Scores the fast profile (RTMPose-m 256x192), the accurate profile (RTMPose-x 384x288 with
flip test) and the accurate profile plus optical-flow wrist refinement, all on the same
held-out deliveries and the same release crops:

- release timing of the rule-based detector (it reads the wrist, so it shows pose quality),
- bowling-arm detection accuracy,
- wrist jitter around release (frame-to-frame acceleration, % of body height),
- wrist dropouts around release and mean joint confidence.

Run (after extracting both caches with ``train.extract_poses`` and ``--accurate``):
``ml/.venv/Scripts/python -m train.compare_pose_models --dataset C:/dev/dataset``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import REPO_ROOT, load_settings
from bowlsense.errors import BowlSenseError
from bowlsense.geometry.angles import keypoint_track
from bowlsense.geometry.body import detect_bowling_arm, joint
from bowlsense.local_settings import accuracy_profile
from bowlsense.phases.heuristic import find_release
from bowlsense.phases.model import load_phase_model
from bowlsense.pose.refine import refine_wrist
from bowlsense.schemas import PoseSequence
from train.dataset import dataset_dir
from train.extract_poses import DEFAULT_OUT, ExtractedDelivery, load_extracted
from train.train_release import heuristic_release, prepare, summarise

log = logging.getLogger("train.compare")
OUT = REPO_ROOT / "data" / "eval" / "pose_compare.json"
NEAR_RELEASE_S = 0.3


def _refined(item: ExtractedDelivery, root: Path) -> ExtractedDelivery:
    """The item with the bowling wrist refined by optical flow around the release the
    rule-based detector finds (never the labelled one, which would leak the answer)."""
    s = accuracy_profile(load_settings())
    seq = item.sequence
    try:
        from bowlsense.pose.smoothing import clean_sequence

        clean = clean_sequence(seq, s.pose)
        arm = detect_bowling_arm(clean).arm
        wrist = keypoint_track(clean, joint(arm, "bowling", "wrist"))
        shoulder = keypoint_track(clean, joint(arm, "bowling", "shoulder"))
        rel = find_release(wrist, shoulder, clean.fps, s.phases.release_refine_frames)
    except BowlSenseError:
        return item
    if rel is None:
        return item
    shifted = seq.model_copy(
        update={
            "bowling_arm": arm,
            "frames": [
                f.model_copy(update={"frame_idx": f.frame_idx + item.window_start})
                for f in seq.frames
            ],
        }
    )
    video = root / "videos" / f"{item.delivery.video}.mp4"
    out = refine_wrist(video, shifted, [rel + item.window_start], s.pose)
    back = out.model_copy(
        update={
            "frames": [
                f.model_copy(update={"frame_idx": f.frame_idx - item.window_start})
                for f in out.frames
            ]
        }
    )
    return item.model_copy(update={"sequence": back})


def _wrist_quality(item: ExtractedDelivery) -> tuple[float, float, float]:
    """(jitter % height, dropout fraction, mean confidence) of the labelled bowling wrist
    around the labelled release, from the raw (unsmoothed) keypoints."""
    seq: PoseSequence = item.sequence
    hand = item.delivery.bowler_hand or "right"
    name = joint(hand, "bowling", "wrist")
    n = round(NEAR_RELEASE_S * seq.fps)
    lo, hi = max(0, item.release_idx - n), min(len(seq.frames), item.release_idx + n + 1)
    xy = np.full((hi - lo, 2), np.nan)
    confs = []
    for i in range(lo, hi):
        k = seq.frames[i].keypoints.get(name)
        if k is not None and k.conf >= 0.3:
            xy[i - lo] = (k.x, k.y)
        confs.append(0.0 if k is None else k.conf)
    heights = [
        abs(f.keypoints["l_ankle"].y - f.keypoints["nose"].y)  # type: ignore[union-attr]
        for f in seq.frames
        if f.keypoints.get("nose") and f.keypoints.get("l_ankle")
    ]
    h = float(np.percentile(heights, 80)) if heights else float(seq.height)
    acc = np.linalg.norm(np.diff(xy, n=2, axis=0), axis=1)
    jitter = float(np.nanmedian(acc) / h * 100) if np.isfinite(acc).any() else float("nan")
    return jitter, float(np.mean(np.isnan(xy[:, 0]))), float(np.mean(confs))


def score(items: list[ExtractedDelivery], model_file: str | None = None) -> dict[str, Any]:
    """Every metric for one set of extracted poses."""
    samples = prepare(items)
    by_key = {s.key: s for s in samples}
    fps = samples[0].seq.fps if samples else 25.0
    errors: list[int | None] = []
    learned: list[int | None] = []
    st = load_settings()
    model = load_phase_model(st.storage.models_path / (model_file or st.phases.model_file))
    rng = np.random.default_rng(0)
    for s in samples:
        n = len(s.seq.frames)
        for _ in range(3):  # three fixed 3 s crops per delivery, as in train_release
            length = min(n, round(3.0 * s.seq.fps))
            lo = max(0, s.release - (length - 4))
            hi = min(s.release - 3, n - length)
            a = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(s.release - length // 2,
                                                                          n - length))  # fmt: skip
            r = heuristic_release(s.seq, a, a + length)
            errors.append(None if r is None else r - (s.release - a))
            crop = s.seq.model_copy(update={"frames": s.seq.frames[a : a + length]})
            try:
                learned.append(model.predict(crop).release - (s.release - a))
            except BowlSenseError:
                learned.append(None)
    arm_hits = [
        detect_bowling_arm(by_key[it.delivery.key].seq).arm == it.delivery.bowler_hand
        for it in items
        if it.delivery.key in by_key and it.delivery.bowler_hand
    ]
    quality = np.array([_wrist_quality(it) for it in items])
    return {
        "deliveries": len(samples),
        "release_rules": summarise(errors, fps),
        "release_learned": summarise(learned, fps),
        "bowling_arm_accuracy": float(np.mean(arm_hits)) if arm_hits else float("nan"),
        "wrist_jitter_pct_height": float(np.nanmean(quality[:, 0])),
        "wrist_dropout_near_release": float(np.mean(quality[:, 1])),
        "wrist_conf_near_release": float(np.mean(quality[:, 2])),
    }


def main(argv: list[str] | None = None) -> int:
    """Score both caches on the shared test deliveries and write data/eval/pose_compare.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--fast", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--accurate", type=Path, default=DEFAULT_OUT.parent / "poses_accurate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = dataset_dir(args.dataset)
    fast = {i.delivery.key: i for i in load_extracted(args.fast) if i.delivery.split == "test"}
    acc = {i.delivery.key: i for i in load_extracted(args.accurate) if i.delivery.split == "test"}
    keys = sorted(fast.keys() & acc.keys())
    log.info("%d shared test deliveries", len(keys))
    st = load_settings()
    acc_model = (
        st.phases.accurate_model_file
        if (st.storage.models_path / st.phases.accurate_model_file).is_file()
        else st.phases.model_file
    )
    results = {
        "fast (RTMPose-m 256x192)": score([fast[k] for k in keys]),
        "accurate (RTMPose-x 384x288 + flip)": score([acc[k] for k in keys], acc_model),
        "accurate + wrist refinement": score([_refined(acc[k], root) for k in keys], acc_model),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, r in results.items():
        rel, lr = r["release_rules"], r["release_learned"]
        log.info(
            "%-38s learned release +-1: %5.1f%% MAE %5.1f ms | rules +-1: %5.1f%% | arm %5.1f%% "
            "| jitter %.2f%% | dropouts %4.1f%% | conf %.2f",
            name, 100 * lr["within_1"], lr["mae_ms"], 100 * rel["within_1"],
            100 * r["bowling_arm_accuracy"], r["wrist_jitter_pct_height"],
            100 * r["wrist_dropout_near_release"], r["wrist_conf_near_release"],
        )  # fmt: skip
    log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
