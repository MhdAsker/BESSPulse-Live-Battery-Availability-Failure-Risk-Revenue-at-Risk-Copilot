"""Transparent engineering-threshold anomaly detector."""

import pandas as pd

from models.anomaly.common import interval_output
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import AnomalyDataset

THRESHOLD_ORIGINS = {
    "temperature_mean_c": "SIMULATOR PHYSICAL LIMIT",
    "temperature_vs_ambient_c": "PROJECT ENGINEERING ASSUMPTION",
    "temperature_spread_c": "PROJECT ENGINEERING ASSUMPTION",
    "temperature_ramp_c": "PROJECT ENGINEERING ASSUMPTION",
    "voltage_spread_v": "PROJECT ENGINEERING ASSUMPTION",
    "rte": "PROJECT ENGINEERING ASSUMPTION",
    "absolute_rack_power_residual_mw": "PROJECT ENGINEERING ASSUMPTION",
    "availability_rate": "SIMULATOR PHYSICAL LIMIT",
}


def score_engineering(dataset: AnomalyDataset, config: AnomalyConfig | None = None) -> pd.DataFrame:
    cfg = config or AnomalyConfig()
    X = dataset.predictors
    rules: list[tuple[str, pd.Series]] = []
    definitions = [
        ("temperature_mean_c", cfg.engineering_temperature_c, "high"),
        ("temperature_vs_ambient_c", cfg.engineering_temperature_vs_ambient_c, "high"),
        ("temperature_spread_c", cfg.engineering_temperature_spread_c, "high"),
        ("temperature_ramp_c", cfg.engineering_temperature_ramp_c, "absolute"),
        ("voltage_spread_v", cfg.engineering_voltage_spread_v, "high"),
        ("rte", cfg.engineering_min_rte, "low"),
        (
            "absolute_rack_power_residual_mw",
            cfg.engineering_power_tracking_error_mw,
            "high",
        ),
    ]
    normalized: list[pd.Series] = []
    for column, threshold, direction in definitions:
        if column not in X:
            continue
        values = X[column]
        if direction == "low":
            severity = (threshold - values).clip(lower=0) / max(threshold, 1e-9)
        elif direction == "absolute":
            severity = values.abs() / threshold
        else:
            severity = values / threshold
        normalized.append(severity)
        rules.append((column, severity.ge(1)))
    if "availability_rate" in X:
        unavailable = (1 - X["availability_rate"]).clip(lower=0)
        normalized.append(unavailable)
        rules.append(("availability_rate", unavailable.gt(0)))
    if not normalized:
        raise ValueError("Engineering detector has no configured signals")
    matrix = pd.concat(normalized, axis=1)
    score = matrix.max(axis=1, skipna=True).fillna(0.0)
    supporting = [
        [name for name, flag in rules if bool(flag.iloc[index])]
        for index in range(len(dataset.keys))
    ]
    output = interval_output(
        dataset.keys,
        score,
        detector_name="engineering",
        detector_version="engineering_v1",
        supporting_signals=supporting,
    )
    output["threshold_origin"] = "PER_RULE; see threshold_origins"
    output["triggered_rules"] = supporting
    output["threshold_origins"] = [
        [THRESHOLD_ORIGINS[name] for name in names] for names in supporting
    ]
    return output
