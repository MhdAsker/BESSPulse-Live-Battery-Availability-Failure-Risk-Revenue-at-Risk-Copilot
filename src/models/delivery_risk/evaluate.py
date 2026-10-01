"""Delivery-risk discrimination, calibration, subgroup, and consistency metrics."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def reliability_bins(
    actual: pd.Series | NDArray[np.int_],
    probability: NDArray[np.float64],
    bins: int = 10,
) -> list[dict[str, float | int]]:
    y = np.asarray(actual, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indexes = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    output: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = indexes == index
        if not mask.any():
            continue
        output.append(
            {
                "bin": index,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return output


def expected_calibration_error(
    actual: pd.Series | NDArray[np.int_], probability: NDArray[np.float64], bins: int = 10
) -> float:
    entries = reliability_bins(actual, probability, bins)
    total = sum(int(entry["count"]) for entry in entries)
    if total == 0:
        return float("nan")
    return float(
        sum(
            int(entry["count"])
            * abs(float(entry["mean_probability"]) - float(entry["observed_frequency"]))
            for entry in entries
        )
        / total
    )


def classification_metrics(
    actual: pd.Series | NDArray[np.int_],
    probability: NDArray[np.float64],
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    y = np.asarray(actual, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("Metrics require equally sized, non-empty actual and probability arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    predicted = p >= threshold
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc_average_precision": (
            float(average_precision_score(y, p)) if y.sum() else float("nan")
        ),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": expected_calibration_error(y, p, calibration_bins),
        "positive_prevalence": float(y.mean()),
        "confusion_matrix": matrix.tolist(),
        "threshold": threshold,
        "row_count": len(y),
    }


def horizon_consistency(predictions: pd.DataFrame) -> dict[str, float]:
    required = ["failure_probability_6h", "failure_probability_12h", "failure_probability_24h"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"Missing horizon probabilities: {missing}")
    valid = predictions[required].dropna()
    if valid.empty:
        return {
            "rows_compared": 0.0,
            "fraction_6h_gt_12h": float("nan"),
            "fraction_12h_gt_24h": float("nan"),
        }
    return {
        "rows_compared": float(len(valid)),
        "fraction_6h_gt_12h": float((valid[required[0]] > valid[required[1]]).mean()),
        "fraction_12h_gt_24h": float((valid[required[1]] > valid[required[2]]).mean()),
    }


def grouped_diagnostics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    probability_column: str,
    minimum_rows: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_name, group in frame.groupby(group_column, dropna=False, observed=True):
        if len(group) < minimum_rows:
            continue
        target = group[target_column].astype(int)
        probability = group[probability_column].to_numpy(dtype=float)
        predicted = probability >= 0.5
        output.append(
            {
                "group": str(group_name),
                "sample_count": len(group),
                "failure_prevalence": float(target.mean()),
                "average_predicted_probability": float(probability.mean()),
                "recall": float(recall_score(target, predicted, zero_division=0)),
            }
        )
    return output
