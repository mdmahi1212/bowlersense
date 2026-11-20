"""Train the pace-vs-spin classifier and evaluate it on the dataset's test split.

Model selection (regularisation strength) uses grouped cross-validation on the train split,
grouped by video so one match's deliveries never sit on both sides. The test split is used
once. Reported: at the labelled release, and end-to-end with the release predicted by the
trained phase model (what the live pipeline would see).

Run: ml/.venv/Scripts/python -m train.train_delivery_type
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from bowlsense.analysis.delivery_type import FEATURES, DeliveryTypeModel, delivery_features
from bowlsense.config import load_settings
from bowlsense.phases.model import load_phase_model
from train.extract_poses import DEFAULT_OUT, load_extracted
from train.train_release import REPORTS, Sample, prepare

log = logging.getLogger("train.delivery_type")


def matrix(samples: list[Sample], releases: list[int]) -> np.ndarray:
    """Feature matrix, one row per sample."""
    return np.array([delivery_features(s.seq, r) for s, r in zip(samples, releases, strict=True)])


def impute(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Replace NaN with the training means."""
    return np.where(np.isfinite(x), x, means[None, :])


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Accuracy, balanced accuracy, F1 for pace."""
    return {
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_fast": float(f1_score(y, pred, pos_label=1)),
        "n": float(len(y)),
    }


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate, save ``models/delivery_type.json`` and a report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out", type=Path, default=load_settings().storage.models_path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    items = load_extracted(args.data)
    labelled = {i.delivery.key: i for i in items if i.delivery.bowling_type is not None}
    train_s = prepare([i for i in labelled.values() if i.delivery.split == "train"])
    test_s = prepare([i for i in labelled.values() if i.delivery.split == "test"])

    def labels(samples: list[Sample]) -> np.ndarray:
        return np.array([labelled[s.key].delivery.bowling_type == "fast" for s in samples], int)

    y_tr, y_te = labels(train_s), labels(test_s)
    x_tr = matrix(train_s, [s.release for s in train_s])
    means = np.nanmean(x_tr, axis=0)
    x_tr = impute(x_tr, means)
    scales = x_tr.std(axis=0)
    scales[scales == 0] = 1.0
    z_tr = (x_tr - means) / scales
    groups = np.array([s.video for s in train_s])

    best_c, best_cv = 1.0, -1.0
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        preds = np.zeros_like(y_tr)
        for tr, va in GroupKFold(n_splits=5).split(z_tr, y_tr, groups):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            preds[va] = clf.fit(z_tr[tr], y_tr[tr]).predict(z_tr[va])
        cv = balanced_accuracy_score(y_tr, preds)
        log.info("C=%-5g grouped-CV balanced accuracy %.3f", c, cv)
        if cv > best_cv:
            best_c, best_cv = c, cv
    clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(z_tr, y_tr)
    model = DeliveryTypeModel(
        feature_names=list(FEATURES),
        means=means.tolist(),
        scales=scales.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
    )

    def evaluate(releases: list[int]) -> tuple[dict[str, float], list[list[int]]]:
        pred = np.array([model.p_fast(row) >= 0.5 for row in matrix(test_s, releases)], int)
        return scores(y_te, pred), confusion_matrix(y_te, pred, labels=[0, 1]).tolist()

    at_label, cm_label = evaluate([s.release for s in test_s])
    phase_model = load_phase_model(args.out / "phase_model.pt")
    predicted = [phase_model.predict(s.seq).release for s in test_s]
    end_to_end, cm_e2e = evaluate(predicted)
    majority = round(float(y_tr.mean()))
    baseline = scores(y_te, np.full_like(y_te, majority))
    model = model.model_copy(update={"metrics": {f"test_{k}": v for k, v in end_to_end.items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "delivery_type.json").write_text(model.model_dump_json(indent=2), encoding="utf-8")
    report: dict[str, Any] = {
        "chosen_C": best_c,
        "grouped_cv_balanced_accuracy": best_cv,
        "test_at_labelled_release": at_label,
        "confusion_matrix_rows_true_cols_pred_order_spin_fast": "see below",
        "test_confusion_at_labelled_release": cm_label,
        "test_end_to_end": end_to_end,
        "test_confusion_end_to_end": cm_e2e,
        "baseline_majority_class": {"predicts": "fast" if majority else "spin", **baseline},
        "coefficients": dict(zip(FEATURES, model.coef, strict=True)),
        "n_train": len(train_s),
        "n_test": len(test_s),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "delivery_type.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, m in (
        ("labelled release", at_label),
        ("end-to-end", end_to_end),
        ("majority baseline", baseline),
    ):
        log.info("%-18s accuracy %.1f%%  balanced %.1f%%  F1(pace) %.2f", name,
                 100 * m["accuracy"], 100 * m["balanced_accuracy"], m["f1_fast"])  # fmt: skip
    log.info("coefficients: %s", {k: round(v, 2) for k, v in report["coefficients"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
