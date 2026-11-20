"""Score the existing rule-based pipeline on real footage against the dataset labels:
bowling-arm detection vs bowler hand, and camera-view estimation vs bowling direction.
(The rule-based release detector is scored in ``train.train_release``.)

Run: ml/.venv/Scripts/python -m train.evaluate_pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from bowlsense.config import load_settings
from bowlsense.errors import BowlSenseError
from bowlsense.geometry.body import detect_bowling_arm
from bowlsense.io.quality import estimate_view
from bowlsense.pose.smoothing import clean_sequence
from train.extract_poses import DEFAULT_OUT, load_extracted
from train.train_release import REPORTS

log = logging.getLogger("train.evaluate")
VIEW_OF_DIRECTION = {
    "towards camera": "front",
    "away from camera": "behind",
    "left-to-right": "side",
    "right-to-left": "side",
}


def main(argv: list[str] | None = None) -> int:
    """Print and save the scores."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    arm_hits, arm_n, arm_fail = 0, 0, 0
    arm_by_hand: Counter[tuple[str, str]] = Counter()
    view_conf: Counter[tuple[str, str]] = Counter()
    for item in load_extracted(args.data):
        d = item.delivery
        seq = clean_sequence(item.sequence, settings.pose)
        try:
            arm = detect_bowling_arm(seq).arm
        except BowlSenseError:
            arm_fail += 1
            continue
        if d.bowler_hand is not None:
            arm_n += 1
            arm_hits += arm == d.bowler_hand
            arm_by_hand[(d.bowler_hand, arm)] += 1
        truth = VIEW_OF_DIRECTION.get(d.bowl_direction or "")
        if truth is not None:
            seq = seq.model_copy(update={"bowling_arm": arm})
            stop = max(2, item.release_idx - round(settings.body.runup_end_before_peak_s * seq.fps))
            view_conf[(truth, estimate_view(seq, (0, stop), settings.video))] += 1
    view_n = sum(view_conf.values())
    view_hits = sum(v for (t, e), v in view_conf.items() if t == e)
    report = {
        "bowling_arm": {
            "accuracy": arm_hits / arm_n if arm_n else None,
            "n": arm_n,
            "wrist_never_seen": arm_fail,
            "by_true_hand(true, predicted)": {f"{t}->{p}": n for (t, p), n in arm_by_hand.items()},
        },
        "camera_view": {
            "accuracy": view_hits / view_n if view_n else None,
            "n": view_n,
            "confusion(true, predicted)": {
                f"{t}->{p}": n for (t, p), n in sorted(view_conf.items())
            },
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "pipeline_on_real_footage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
