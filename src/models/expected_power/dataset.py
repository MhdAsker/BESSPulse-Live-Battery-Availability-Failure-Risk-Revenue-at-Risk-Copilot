"""Leakage-safe expected-power dataset construction."""

from typing import Any

import pandas as pd

from features.validation import prepare_temporal_frame, validate_no_leakage
from models.common import ModelDataset, dataset_hash
from models.expected_power.config import ExpectedPowerConfig

POWER_NUMERIC_FEATURES = (
    "requested_power_mw",
    "abs_requested_power_mw",
    "soc",
    "ambient_temperature_c",
    "available_racks",
    "available_pcs",
    "available_rack_fraction",
    "available_pcs_fraction",
    "available_power_mw",
    "recent_alarm_count",
    "alarm_frequency",
    "rte",
)
POWER_CATEGORICAL_FEATURES = ("operating_mode",)
POWER_TARGET = "actual_power_mw"


def build_expected_power_dataset(
    site_features: pd.DataFrame,
    config: ExpectedPowerConfig | None = None,
) -> ModelDataset:
    """Use truth only for an offline healthy-row mask, never as X."""

    cfg = config or ExpectedPowerConfig()
    required = [
        "timestamp_utc",
        "asset_id",
        POWER_TARGET,
        "requested_power_mw",
        "soc",
        "available_racks",
        "available_pcs",
        "available_rack_fraction",
        "available_pcs_fraction",
        "available_power_mw",
        "recent_alarm_count",
        "alarm_frequency",
        "rte",
        "operating_mode",
    ]
    missing = [column for column in required if column not in site_features]
    if missing:
        raise ValueError(f"Missing expected-power data columns: {missing}")
    frame = site_features.copy()
    if "ambient_temperature_c" not in frame:
        frame["ambient_temperature_c"] = pd.NA
    frame["abs_requested_power_mw"] = frame["requested_power_mw"].abs()
    frame = prepare_temporal_frame(frame, ["asset_id"])
    predictors = frame[[*POWER_NUMERIC_FEATURES, *POWER_CATEGORICAL_FEATURES]].copy()
    validate_no_leakage(list(predictors.columns))
    completely_missing = [column for column in predictors if predictors[column].isna().all()]
    if completely_missing:
        raise ValueError(f"Predictors are completely missing: {completely_missing}")
    if cfg.require_healthy_labels and "ground_truth_label" not in frame:
        raise ValueError("Offline expected-power training requires ground_truth_label")
    labels = frame.get("ground_truth_label", pd.Series("NORMAL", index=frame.index)).astype(str)
    healthy = labels.isin(cfg.healthy_labels)
    healthy &= frame["available_racks"].gt(0) & frame["available_pcs"].gt(0)
    if cfg.active_intervals_only_for_training:
        healthy &= frame["requested_power_mw"].abs().gt(cfg.minimum_request_threshold_mw)
    keys = frame[["timestamp_utc", "asset_id"]].copy()
    target = pd.to_numeric(frame[POWER_TARGET], errors="coerce")
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
        target_name=POWER_TARGET,
        dataset_hash=dataset_hash(keys, predictors, target),
    )


def power_dataset_coverage(dataset: ModelDataset) -> dict[str, Any]:
    return {
        "row_count": len(dataset.target),
        "start": dataset.keys["timestamp_utc"].min(),
        "end": dataset.keys["timestamp_utc"].max(),
        "healthy_rows": int(dataset.healthy.sum()),
        "fault_or_excluded_rows": int((~dataset.healthy).sum()),
        "missing_predictors": dataset.predictors.isna().sum().to_dict(),
        "target_min": float(dataset.target.min()),
        "target_max": float(dataset.target.max()),
        "target_mean": float(dataset.target.mean()),
        "target_std": float(dataset.target.std()),
        "operating_mode_distribution": dataset.predictors["operating_mode"]
        .value_counts()
        .to_dict(),
    }
