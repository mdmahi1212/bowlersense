"""Train the bowler selector on labelled match footage.

Input: the cached tracks per delivery from ``train.eval_bowler`` (data/training/tracks).
A track is a positive example if it matches the labelled bowler (IoU >= 0.5 in at least half
of the labelled frames). A logistic regression (balanced classes, standardised features) is
fitted on the train split, the regularisation strength picked by grouped cross-validation
on train videos, then evaluated on the held-out test videos as *selection accuracy*:
how often the top-scoring track is the bowler. The hand-written rules are scored on the
same deliveries for comparison. The model is saved only if it beats the old rule on test.

Run: ml/.venv/Scripts/python -m train.train_bowler_select
Writes models/bowler_selector.json and data/training/reports/bowler_selector.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import load_settings
from bowlsense.detect.person import PersonTrack, select_bowler
from bowlsense.detect.selector import FEATURE_VERSION, FEATURES, feature_matrix
from bowlsense.errors import NoBowlerDetectedError
from train.eval_bowler import MIN_MATCH_FRACTION, TRACK_CACHE, legacy_select, match_fraction
from train.train_release import REPORTS

log = logging.getLogger("train.bowler_select")


class Clip:
    """One delivery's tracks, labels and per-track targets."""

    def __init__(self, data: dict[str, Any], min_frames: int) -> None:
        self.video: str = data["video"]
        self.split: str = data["split"]
        self.width, self.height = int(data["width"]), int(data["height"])
        self.n_frames = int(data["hi"]) - int(data["lo"])
        labels = {int(k): tuple(v) for k, v in data["labels"].items()}
        all_tracks = [PersonTrack.model_validate(t) for t in data["tracks"]]
        self.tracks = [t for t in all_tracks if len(t.frames) >= min_frames]
        self.match = [match_fraction(t, labels) for t in self.tracks]
        self.labels = labels

    @property
    def has_bowler(self) -> bool:
        return any(m >= MIN_MATCH_FRACTION for m in self.match)

    def features(self) -> np.ndarray:
        return feature_matrix(self.tracks, self.width, self.height, self.n_frames)

    def target(self) -> np.ndarray:
        return np.array([m >= MIN_MATCH_FRACTION for m in self.match], dtype=int)


def load_clips(min_frames: int) -> list[Clip]:
    """Every cached delivery with at least one usable track."""
    clips = [
        Clip(json.loads(p.read_text(encoding="utf-8")), min_frames)
        for p in sorted(TRACK_CACHE.glob("*.json"))
    ]
    return [c for c in clips if c.tracks]


def fit(clips: list[Clip], c_value: float) -> Any:
    """Standardise + logistic regression on every track of ``clips``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([c.features() for c in clips])
    y = np.concatenate([c.target() for c in clips])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000)
    )
    model.fit(x, y)
    return model


def selection_accuracy(model: Any, clips: list[Clip]) -> float:
    """Share of deliveries whose top-scoring track is the bowler."""
    hits = [c.match[int(np.argmax(model.predict_proba(c.features())[:, 1]))] >= MIN_MATCH_FRACTION
            for c in clips]  # fmt: skip
    return float(np.mean(hits)) if hits else float("nan")


def rule_accuracy(clips: list[Clip], min_frames: int) -> dict[str, float]:
    """The hand-written rules on the same deliveries."""
    out: dict[str, list[bool]] = {"legacy": [], "rule_in_app": []}
    for c in clips:
        for name, pick in (("legacy", legacy_select), ("rule_in_app", select_bowler)):
            try:
                chosen = pick(c.tracks, c.width, min_frames)
                hit = match_fraction(chosen, c.labels) >= MIN_MATCH_FRACTION
            except NoBowlerDetectedError:
                hit = False
            out[name].append(hit)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    """Pick C by grouped CV on train videos, fit, evaluate on test, save if better."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="model path (default: models/bowler_selector.json)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = load_settings()
    min_frames = s.detect.min_track_frames
    clips = load_clips(min_frames)
    train = [c for c in clips if c.split == "train" and c.has_bowler]
    test = [c for c in clips if c.split == "test"]
    test_trackable = [c for c in test if c.has_bowler]
    videos = sorted({c.video for c in train})
    folds = [videos[i::4] for i in range(4)]
    best_c, best_cv = 1.0, -1.0
    for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        accs = []
        for hold in folds:
            fit_on = [c for c in train if c.video not in hold]
            val = [c for c in train if c.video in hold]
            accs.append(selection_accuracy(fit(fit_on, c_value), val))
        cv = float(np.mean(accs))
        log.info("C=%-5g grouped-CV selection accuracy %.3f", c_value, cv)
        if cv > best_cv:
            best_c, best_cv = c_value, cv
    model = fit(train, best_c)
    learned_all = selection_accuracy(model, test)
    learned_trackable = selection_accuracy(model, test_trackable)
    rules_all = rule_accuracy(test, min_frames)
    rules_trackable = rule_accuracy(test_trackable, min_frames)
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    report = {
        "train_deliveries": len(train),
        "test_deliveries": len(test),
        "test_trackable": len(test_trackable),
        "chosen_C": best_c,
        "cv_selection_accuracy": best_cv,
        "test_selection_accuracy": {
            "learned": learned_all,
            **rules_all,
            "tracker_limit": len(test_trackable) / len(test) if test else None,
        },
        "test_selection_accuracy_when_tracked": {"learned": learned_trackable, **rules_trackable},
        "weights": dict(zip(FEATURES, (float(w) for w in lr.coef_[0]), strict=True)),
    }
    better = learned_all > rules_all["legacy"]
    report["saved"] = better
    if better:
        out = args.out or s.storage.models_path / s.detect.selector_model
        out.write_text(json.dumps({
            "feature_version": FEATURE_VERSION, "features": list(FEATURES),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "trained_on": f"{len(train)} labelled deliveries (train split)",
            "test_selection_accuracy": learned_all,
        }, indent=1), encoding="utf-8")  # fmt: skip
        log.info("saved %s", out)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "bowler_selector.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(json.dumps({k: v for k, v in report.items() if k != "weights"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
