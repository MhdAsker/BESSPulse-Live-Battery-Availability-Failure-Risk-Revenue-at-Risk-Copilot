"""Prediction distributions and mature-label classification metrics."""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_score, recall_score

from monitoring.metrics import brier_score


def prediction_distribution(
    probabilities: NDArray[np.float64], threshold: float = 0.5
) -> dict[str, float | None]:
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "p95": None, "high_risk_fraction": None}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "high_risk_fraction": float(np.mean(values >= threshold)),
    }


def classification_performance(
    probabilities: NDArray[np.float64], labels: NDArray[np.float64], threshold: float = 0.5
) -> dict[str, float | None]:
    if not len(labels) or len(probabilities) != len(labels):
        return {"brier": None, "precision": None, "recall": None, "pr_auc": None}
    predicted = probabilities >= threshold
    truth = labels.astype(int)
    has_both_classes = len(np.unique(truth)) == 2
    return {
        "brier": brier_score(probabilities, labels),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, probabilities))
        if has_both_classes
        else None,
    }
