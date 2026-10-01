"""Regression and price-regime evaluation."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score


def price_metrics(actual: pd.Series, prediction: NDArray[np.float64]) -> dict[str, float]:
    truth = actual.to_numpy(dtype=float)
    residual = prediction - truth
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "mean_bias_prediction_minus_actual": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "p95_absolute_error": float(np.percentile(np.abs(residual), 95)),
    }


def regime_metrics(
    actual: pd.Series, prediction: NDArray[np.float64], high_price_threshold: float
) -> dict[str, object]:
    truth = actual.to_numpy(dtype=float)
    result: dict[str, object] = {"high_price_threshold_eur_per_mwh": high_price_threshold}
    for name, mask in (
        ("negative", truth < 0),
        ("high", truth >= high_price_threshold),
    ):
        count = int(mask.sum())
        result[f"{name}_count"] = count
        result[f"{name}_mae"] = (
            float(mean_absolute_error(truth[mask], prediction[mask])) if count else None
        )
        result[f"{name}_bias_prediction_minus_actual"] = (
            float(np.mean(prediction[mask] - truth[mask])) if count else None
        )
    return result


def quantile_metrics(
    actual: pd.Series, raw_predictions: dict[float, NDArray[np.float64]]
) -> dict[str, object]:
    truth = actual.to_numpy(dtype=float)
    levels = sorted(raw_predictions)
    matrix = np.column_stack([raw_predictions[level] for level in levels])
    result: dict[str, object] = {}
    for index, level in enumerate(levels):
        prediction = matrix[:, index]
        result[f"p{int(level * 100):02d}_pinball_loss"] = float(
            mean_pinball_loss(truth, prediction, alpha=level)
        )
        result[f"p{int(level * 100):02d}_empirical_coverage"] = float(np.mean(truth <= prediction))
    result["p10_gt_p50_fraction"] = float(np.mean(matrix[:, 0] > matrix[:, 1]))
    result["p50_gt_p90_fraction"] = float(np.mean(matrix[:, 1] > matrix[:, 2]))
    return result
