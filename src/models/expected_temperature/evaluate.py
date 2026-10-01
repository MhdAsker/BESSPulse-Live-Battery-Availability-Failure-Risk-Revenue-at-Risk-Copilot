"""Expected-temperature metrics and diagnostics."""

import pandas as pd

from models.common import regression_metrics


def thermal_residual_fault_diagnostic(
    residual_frame: pd.DataFrame, label_column: str = "ground_truth_label"
) -> dict[str, float]:
    if label_column not in residual_frame:
        return {}
    normal = residual_frame.loc[
        residual_frame[label_column].astype(str).str.upper().eq("NORMAL"),
        "thermal_residual_c",
    ].dropna()
    fault = residual_frame.loc[
        ~residual_frame[label_column].astype(str).str.upper().eq("NORMAL"),
        "thermal_residual_c",
    ].dropna()
    if normal.empty or fault.empty:
        return {}
    healthy_p95 = float(normal.abs().quantile(0.95))
    return {
        "normal_residual_median_c": float(normal.median()),
        "fault_residual_median_c": float(fault.median()),
        "normal_absolute_residual_p95_c": healthy_p95,
        "fault_absolute_residual_median_c": float(fault.abs().median()),
        "fault_fraction_exceeding_healthy_p95": float((fault.abs() > healthy_p95).mean()),
    }


def grouped_temperature_metrics(
    frame: pd.DataFrame, group_column: str
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group, rows in frame.groupby(group_column):
        output[str(group)] = regression_metrics(
            rows["temperature_mean_c"], rows["expected_temperature_c"].to_numpy()
        )
    return output
