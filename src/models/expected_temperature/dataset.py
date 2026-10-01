"""Cross-rack causal expected-temperature dataset construction."""

from typing import Any

import pandas as pd

from features.validation import (
    exact_time_lag,
    prepare_temporal_frame,
    trailing_statistic,
    validate_no_leakage,
)
from models.common import ModelDataset, dataset_hash
from models.expected_temperature.config import ExpectedTemperatureConfig

TEMPERATURE_NUMERIC_FEATURES = (
    "actual_power_mw",
    "abs_rack_power_mw",
    "requested_power_mw",
    "ambient_temperature_c",
    "soc",
    "temperature_lag_5m",
    "temperature_lag_15m",
    "rolling_temperature_mean_30m",
    "rolling_absolute_power_30m",
    "cooling_state_proxy",
    "throughput_change_mwh",
)
TEMPERATURE_CATEGORICAL_FEATURES = ("operating_state",)
TEMPERATURE_TARGET = "temperature_mean_c"


def build_expected_temperature_dataset(
    rack_features: pd.DataFrame,
    config: ExpectedTemperatureConfig | None = None,
) -> ModelDataset:
    """Build lags per rack; rack/PCS identity remains metadata, never X."""

    cfg = config or ExpectedTemperatureConfig()
    required = [
        "timestamp_utc",
        "rack_id",
        "pcs_id",
        TEMPERATURE_TARGET,
        "actual_power_mw",
        "requested_power_mw",
        "ambient_temperature_c",
        "soc",
        "throughput_change_mwh",
        "availability",
        "operating_state",
    ]
    missing = [column for column in required if column not in rack_features]
    if missing:
        raise ValueError(f"Missing expected-temperature data columns: {missing}")
    frame = prepare_temporal_frame(rack_features.copy(), ["rack_id"])
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("rack_id", sort=False):
        timestamps = group["timestamp_utc"]
        temperature = group[TEMPERATURE_TARGET]
        group["temperature_lag_5m"] = exact_time_lag(
            temperature, timestamps, cfg.temperature_lag_short
        )
        group["temperature_lag_15m"] = exact_time_lag(
            temperature, timestamps, cfg.temperature_lag_long
        )
        group["rolling_temperature_mean_30m"] = trailing_statistic(
            temperature.shift(1),
            timestamps,
            cfg.recent_window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["rolling_absolute_power_30m"] = trailing_statistic(
            group["actual_power_mw"].abs(),
            timestamps,
            cfg.recent_window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["cooling_state_proxy"] = group["temperature_lag_5m"] - group["ambient_temperature_c"]
        pieces.append(group)
    frame = pd.concat(pieces, ignore_index=True)
    frame["abs_rack_power_mw"] = frame["actual_power_mw"].abs()
    predictors = frame[[*TEMPERATURE_NUMERIC_FEATURES, *TEMPERATURE_CATEGORICAL_FEATURES]].copy()
    validate_no_leakage(list(predictors.columns))
    if "rack_id" in predictors or "pcs_id" in predictors:
        raise AssertionError("Rack identity entered expected-temperature predictors")
    completely_missing = [column for column in predictors if predictors[column].isna().all()]
    if completely_missing:
        raise ValueError(f"Predictors are completely missing: {completely_missing}")
    if cfg.require_healthy_labels and "ground_truth_label" not in frame:
        raise ValueError("Offline expected-temperature training requires ground_truth_label")
    labels = frame.get("ground_truth_label", pd.Series("NORMAL", index=frame.index)).astype(str)
    healthy = labels.isin(cfg.healthy_labels)
    healthy &= frame["availability"].astype(bool)
    healthy &= frame["temperature_lag_5m"].notna()
    keys = frame[["timestamp_utc", "rack_id", "pcs_id"]].copy()
    target = pd.to_numeric(frame[TEMPERATURE_TARGET], errors="coerce")
    metadata_columns = [
        column for column in ["ground_truth_label", "fault_id", "fault_type"] if column in frame
    ]
    metadata = frame[metadata_columns].copy()
    return ModelDataset(
        keys=keys,
        predictors=predictors,
        target=target,
        healthy=healthy,
        evaluation_metadata=metadata,
        target_name=TEMPERATURE_TARGET,
        dataset_hash=dataset_hash(keys, predictors, target),
    )


def temperature_dataset_coverage(dataset: ModelDataset) -> dict[str, Any]:
    rack_counts = dataset.keys["rack_id"].value_counts()
    return {
        "row_count": len(dataset.target),
        "start": dataset.keys["timestamp_utc"].min(),
        "end": dataset.keys["timestamp_utc"].max(),
        "healthy_rows": int(dataset.healthy.sum()),
        "fault_or_excluded_rows": int((~dataset.healthy).sum()),
        "missing_predictors": dataset.predictors.isna().sum().to_dict(),
        "target_min_c": float(dataset.target.min()),
        "target_max_c": float(dataset.target.max()),
        "target_mean_c": float(dataset.target.mean()),
        "target_std_c": float(dataset.target.std()),
        "rack_count": int(rack_counts.size),
        "rows_per_rack_min": int(rack_counts.min()),
        "rows_per_rack_max": int(rack_counts.max()),
    }
