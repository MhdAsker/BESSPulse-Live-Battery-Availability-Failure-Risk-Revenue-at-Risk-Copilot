"""Leakage-guarded rack anomaly features and chronological event-aware splits."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from features.validation import is_prohibited_column
from models.anomaly.config import AnomalyConfig

ANOMALY_FEATURES = (
    "temperature_vs_ambient_c",
    "temperature_ramp_c",
    "temperature_spread_c",
    "voltage_spread_v",
    "soc",
    "rte",
    "rte_trend",
    "rack_power_residual_mw",
    "absolute_rack_power_residual_mw",
    "alarm_frequency",
    "availability_rate",
    "temperature_peer_deviation",
    "temperature_peer_robust_zscore",
    "voltage_spread_peer_deviation",
    "voltage_spread_peer_robust_zscore",
    "rte_peer_deviation",
    "rte_peer_robust_zscore",
    "soc_peer_deviation",
    "soc_peer_robust_zscore",
    "power_tracking_peer_deviation",
    "power_tracking_peer_robust_zscore",
    "peer_count",
    "expected_temperature_c",
    "thermal_residual_c",
    "thermal_residual_rolling_mean",
    "positive_thermal_residual_fraction",
    "thermal_residual_persistence_count",
    "delivery_shortfall_mw",
)


def is_anomaly_leakage(column: str) -> bool:
    lowered = column.lower()
    return (
        is_prohibited_column(column)
        or lowered in {"severity", "progression_rate", "time_to_fault", "time_to_failure"}
        or lowered.startswith("future_")
        or lowered.startswith("failure_within_")
        or "failure_event_id" in lowered
        or "shap" in lowered
    )


@dataclass(frozen=True)
class AnomalyDataset:
    keys: pd.DataFrame
    predictors: pd.DataFrame
    evaluation_metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    dataset_hash: str

    def subset(self, positions: NDArray[np.intp]) -> "AnomalyDataset":
        return AnomalyDataset(
            self.keys.iloc[positions].reset_index(drop=True),
            self.predictors.iloc[positions].reset_index(drop=True),
            self.evaluation_metadata.iloc[positions].reset_index(drop=True),
            self.feature_names,
            self.dataset_hash,
        )


@dataclass(frozen=True)
class AnomalySplit:
    train: AnomalyDataset
    validation: AnomalyDataset
    test: AnomalyDataset


def build_anomaly_dataset(
    feature_frame: pd.DataFrame,
    evaluation_metadata: pd.DataFrame | None = None,
) -> AnomalyDataset:
    leakage = [column for column in feature_frame if is_anomaly_leakage(str(column))]
    if leakage:
        raise ValueError(f"Anomaly feature input contains leakage fields: {leakage}")
    required = ["timestamp_utc", "rack_id", "pcs_id"]
    missing = [column for column in required if column not in feature_frame]
    if missing:
        raise ValueError(f"Missing anomaly keys: {missing}")
    frame = feature_frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame.duplicated(["timestamp_utc", "rack_id"]).any():
        raise ValueError("Anomaly rows must be unique by timestamp_utc and rack_id")
    selected = tuple(column for column in ANOMALY_FEATURES if column in frame)
    if not selected:
        raise ValueError("No curated anomaly features are available")
    predictors = frame[list(selected)].copy()
    numeric = predictors.apply(pd.to_numeric, errors="coerce")
    incompatible = predictors.notna() & numeric.isna()
    if incompatible.any().any():
        raise ValueError("Anomaly predictors contain incompatible numeric values")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Anomaly predictors contain infinite values")
    keys = frame[required].copy()
    order = np.lexsort((keys["rack_id"].to_numpy(), keys["timestamp_utc"].astype("int64")))
    keys = keys.iloc[order].reset_index(drop=True)
    predictors = numeric.iloc[order].reset_index(drop=True)
    if evaluation_metadata is None:
        metadata = pd.DataFrame(index=keys.index)
    else:
        truth = evaluation_metadata.copy()
        truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True, errors="raise")
        metadata_columns = [column for column in truth if column not in required]
        metadata = keys.merge(truth, on=required, how="left", validate="one_to_one")
        metadata = metadata[metadata_columns]
    combined = pd.concat([keys, predictors], axis=1)
    values = pd.util.hash_pandas_object(combined, index=False).to_numpy().tobytes()
    schema = "|".join(f"{column}:{combined[column].dtype}" for column in combined)
    digest = hashlib.sha256(schema.encode() + values).hexdigest()
    return AnomalyDataset(keys, predictors, metadata.reset_index(drop=True), selected, digest)


def chronological_anomaly_split(dataset: AnomalyDataset, config: AnomalyConfig) -> AnomalySplit:
    timestamps = pd.DatetimeIndex(dataset.keys["timestamp_utc"].unique()).sort_values()
    if len(timestamps) < 6:
        raise ValueError("Too few timestamps for anomaly split")
    cuts = [
        int(len(timestamps) * config.train_fraction),
        int(len(timestamps) * (config.train_fraction + config.validation_fraction)),
    ]
    boundaries = [timestamps[max(1, min(cut, len(timestamps) - 1))] for cut in cuts]
    if "fault_id" in dataset.evaluation_metadata:
        event_frame = pd.DataFrame(
            {
                "timestamp_utc": dataset.keys["timestamp_utc"],
                "fault_id": dataset.evaluation_metadata["fault_id"],
            }
        ).dropna(subset=["fault_id"])
        for index, boundary in enumerate(boundaries):
            for _, event in event_frame.groupby("fault_id"):
                start = event["timestamp_utc"].min()
                end = event["timestamp_utc"].max()
                if start < boundary <= end:
                    boundaries[index] = min(boundaries[index], start)
    time = dataset.keys["timestamp_utc"]
    masks = [
        time < boundaries[0],
        (time >= boundaries[0]) & (time < boundaries[1]),
        time >= boundaries[1],
    ]
    partitions = [dataset.subset(np.flatnonzero(mask.to_numpy())) for mask in masks]
    if any(partition.keys.empty for partition in partitions):
        raise ValueError("Anomaly split produced an empty partition")
    split = AnomalySplit(*partitions)
    if not (
        split.train.keys.timestamp_utc.max()
        < split.validation.keys.timestamp_utc.min()
        <= split.validation.keys.timestamp_utc.max()
        < split.test.keys.timestamp_utc.min()
    ):
        raise AssertionError("Anomaly partitions are not chronological")
    return split


def healthy_mask(dataset: AnomalyDataset) -> pd.Series:
    if "ground_truth_label" not in dataset.evaluation_metadata:
        raise ValueError("Offline healthy-cohort selection requires ground_truth_label")
    return dataset.evaluation_metadata["ground_truth_label"].fillna("NORMAL").eq("NORMAL")


def feature_contract() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "provenance": "MODEL_PREDICTION" if name == "expected_temperature_c" else "DERIVED",
            "causal_status": "available_at_or_before_T",
            "excluded_identity": name not in {"rack_id", "pcs_id"},
        }
        for name in ANOMALY_FEATURES
    ]
