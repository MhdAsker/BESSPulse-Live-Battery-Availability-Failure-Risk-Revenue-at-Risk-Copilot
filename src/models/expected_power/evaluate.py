"""Expected-power operational metrics and fault-period residual diagnostics."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from models.common import regression_metrics


def power_metrics(
    actual: pd.Series,
    prediction: NDArray[np.float64],
    requested: pd.Series,
    threshold: float,
) -> dict[str, float]:
    metrics = regression_metrics(actual, prediction)
    active = requested.abs().gt(threshold).to_numpy()
    if active.any():
        active_metrics = regression_metrics(actual.loc[active], prediction[active])
        metrics["active_mae"] = active_metrics["mae"]
        metrics["active_rmse"] = active_metrics["rmse"]
    else:
        metrics["active_mae"] = float("nan")
        metrics["active_rmse"] = float("nan")
    return metrics


def residual_fault_diagnostic(
    residual_frame: pd.DataFrame, label_column: str = "ground_truth_label"
) -> dict[str, float]:
    if label_column not in residual_frame:
        return {}
    normal = residual_frame.loc[
        residual_frame[label_column].astype(str).str.upper().eq("NORMAL"),
        "power_residual_mw",
    ].dropna()
    fault = residual_frame.loc[
        ~residual_frame[label_column].astype(str).str.upper().eq("NORMAL"),
        "power_residual_mw",
    ].dropna()
    if normal.empty or fault.empty:
        return {}
    healthy_p95 = float(normal.abs().quantile(0.95))
    return {
        "normal_residual_median": float(normal.median()),
        "fault_residual_median": float(fault.median()),
        "normal_absolute_residual_p95": healthy_p95,
        "fault_absolute_residual_median": float(fault.abs().median()),
        "fault_fraction_exceeding_healthy_p95": float((fault.abs() > healthy_p95).mean()),
    }
