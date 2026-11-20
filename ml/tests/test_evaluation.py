"""The evaluation metrics against hand-computed examples."""

from __future__ import annotations

import pytest

from bowlsense.evaluation import classification_report, confusion_matrix, regression_report


def test_confusion_matrix_rows_true_cols_pred() -> None:
    m = confusion_matrix(["a", "a", "b", "b", "b"], ["a", "b", "b", "b", "a"], ["a", "b"])
    assert m.tolist() == [[1, 1], [1, 2]]
    with pytest.raises(ValueError, match="unknown label"):
        confusion_matrix(["a"], ["c"], ["a", "b"])


def test_binary_metrics_hand_computed() -> None:
    # 10 cases: TP=4 FN=1 (positives "y"), TN=3 FP=2
    y_true = ["y"] * 5 + ["n"] * 5
    y_pred = ["y", "y", "y", "y", "n", "n", "n", "n", "y", "y"]
    r = classification_report("t", "d", "synthetic", y_true, y_pred, ["n", "y"])
    assert r.confusion == [[3, 2], [1, 4]]
    assert r.accuracy == pytest.approx(0.7)
    y = r.per_class[1]
    assert y.precision == pytest.approx(4 / 6) and y.recall == pytest.approx(0.8)
    assert y.f1 == pytest.approx(2 * (4 / 6) * 0.8 / (4 / 6 + 0.8))
    n = r.per_class[0]
    assert n.precision == pytest.approx(0.75) and n.recall == pytest.approx(0.6)
    assert r.balanced_accuracy == pytest.approx(0.7)
    assert r.macro_f1 == pytest.approx((n.f1 + y.f1) / 2)  # type: ignore[operator]
    assert r.baseline_accuracy == pytest.approx(0.5)


def test_abstentions_lower_coverage_not_accuracy() -> None:
    r = classification_report(
        "t", "d", "synthetic", ["a", "a", "b", "b"], ["a", None, "b", None], ["a", "b"]
    )
    assert r.n == 2 and r.abstained == 2 and r.coverage == pytest.approx(0.5)
    assert r.accuracy == pytest.approx(1.0)


def test_never_predicted_class_has_no_precision() -> None:
    r = classification_report("t", "d", "real", ["a", "b"], ["a", "a"], ["a", "b"])
    b = r.per_class[1]
    assert b.precision is None and b.recall == 0.0 and b.f1 is None


def test_regression_metrics() -> None:
    r = regression_report("t", "d", "synthetic", "deg", [10, 20, 30], [12, 18, None])
    assert r.n == 2 and r.mae == pytest.approx(2.0) and r.rmse == pytest.approx(2.0)
    assert r.bias == pytest.approx(0.0) and r.max_abs_error == pytest.approx(2.0)
    assert r.r2 == pytest.approx(1 - 8 / 50)
    flat = regression_report("t", "d", "synthetic", "deg", [5, 5], [5, 6])
    assert flat.r2 is None
