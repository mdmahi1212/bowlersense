"""Evaluation metrics: classification (confusion matrix, accuracy, precision, recall, F1) and
measurement accuracy (MAE, RMSE, R^2, bias), plus the report models shown in the web app.

Pure NumPy so the numbers are transparent; ``tests/test_evaluation.py`` checks them against
hand-computed examples.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

DataKind = Literal["real", "synthetic"]


class ClassMetrics(BaseModel):
    """Precision, recall and F1 of one class."""

    label: str
    precision: float | None  # None: the class was never predicted
    recall: float | None  # None: the class never occurs
    f1: float | None
    support: int  # how many true examples of this class


class ClassificationReport(BaseModel):
    """Everything about one classifier on one labelled test set."""

    task: str
    description: str
    data: DataKind
    labels: list[str]
    confusion: list[list[int]]  # rows = true label, columns = predicted label
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: list[ClassMetrics]
    abstained: int = 0  # cases the system declined to classify (e.g. "can't judge")
    coverage: float = 1.0  # share of cases that got a prediction
    baseline_accuracy: float | None = None  # always predicting the most common class
    notes: list[str] = Field(default_factory=list)


class RegressionReport(BaseModel):
    """How close a measurement is to the ground truth."""

    task: str
    description: str
    data: DataKind
    unit: str
    n: int
    mae: float
    rmse: float
    r2: float | None  # None when the truth doesn't vary
    bias: float  # mean (measured - truth)
    max_abs_error: float
    truth: list[float]
    measured: list[float]


class ScoreItem(BaseModel):
    """A single headline score (e.g. a task that isn't a fixed-label classification)."""

    task: str
    metric: str
    value: float
    n: int
    data: DataKind
    note: str = ""


class EvaluationReport(BaseModel):
    """The full evaluation shown on the "Model evaluation" page."""

    generated: datetime
    classification: list[ClassificationReport]
    regression: list[RegressionReport]
    scores: list[ScoreItem]
    notes: list[str] = Field(default_factory=list)


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    """Counts with rows = true label and columns = predicted label (in ``labels`` order)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    index = {label: i for i, label in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in index or p not in index:
            raise ValueError(f"unknown label {t!r} or {p!r}; labels are {list(labels)}")
        m[index[t], index[p]] += 1
    return m


def classification_report(
    task: str,
    description: str,
    data: DataKind,
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    labels: Sequence[str],
    notes: Sequence[str] = (),
) -> ClassificationReport:
    """Metrics for one classifier; ``None`` predictions are abstentions (not in the matrix,
    counted in ``abstained`` / ``coverage``)."""
    kept = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if p is not None]
    if not kept:
        raise ValueError(f"{task}: no predictions to evaluate")
    t_kept = [t for t, _ in kept]
    p_kept = [p for _, p in kept if p is not None]
    m = confusion_matrix(t_kept, p_kept, labels)
    n = int(m.sum())
    per_class = []
    for i, label in enumerate(labels):
        tp = float(m[i, i])
        precision = _div(tp, float(m[:, i].sum()))
        recall = _div(tp, float(m[i, :].sum()))
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else (0.0 if precision is not None and recall is not None else None)
        )
        per_class.append(
            ClassMetrics(
                label=label, precision=precision, recall=recall, f1=f1, support=int(m[i].sum())
            )
        )
    present = [c for c in per_class if c.support > 0]
    supports = np.array([c.support for c in present], dtype=float)

    def mean(values: list[float | None]) -> float:
        vals = [v if v is not None else 0.0 for v in values]
        return float(np.mean(vals)) if vals else 0.0

    counts = np.bincount([list(labels).index(t) for t in y_true], minlength=len(labels))
    return ClassificationReport(
        task=task,
        description=description,
        data=data,
        labels=list(labels),
        confusion=m.tolist(),
        n=n,
        accuracy=float(np.trace(m)) / n,
        balanced_accuracy=mean([c.recall for c in present]),
        macro_precision=mean([c.precision for c in present]),
        macro_recall=mean([c.recall for c in present]),
        macro_f1=mean([c.f1 for c in present]),
        weighted_f1=float(
            np.sum([(c.f1 or 0.0) * w for c, w in zip(present, supports, strict=True)])
            / supports.sum()
        ),
        per_class=per_class,
        abstained=len(y_true) - n,
        coverage=n / len(y_true),
        baseline_accuracy=float(counts.max()) / len(y_true),
        notes=list(notes),
    )


def regression_report(
    task: str,
    description: str,
    data: DataKind,
    unit: str,
    truth: Sequence[float],
    measured: Sequence[float | None],
) -> RegressionReport:
    """MAE, RMSE, R^2 and bias of a measurement against ground truth (``None``s skipped)."""
    pairs = [(float(t), float(m)) for t, m in zip(truth, measured, strict=True)
             if m is not None and np.isfinite(m)]  # fmt: skip
    if not pairs:
        raise ValueError(f"{task}: nothing measured")
    t = np.array([p[0] for p in pairs])
    m = np.array([p[1] for p in pairs])
    err = m - t
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return RegressionReport(
        task=task,
        description=description,
        data=data,
        unit=unit,
        n=len(pairs),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        r2=1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else None,
        bias=float(np.mean(err)),
        max_abs_error=float(np.max(np.abs(err))),
        truth=t.tolist(),
        measured=m.tolist(),
    )
