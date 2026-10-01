"""Curated feature manifest, horizon datasets, and purged chronological splits."""

import hashlib
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from features.validation import is_prohibited_column
from models.delivery_risk.config import DeliveryRiskConfig

FEATURE_MANIFEST: tuple[dict[str, str], ...] = (
    {"name": "requested_power_mw", "family": "base", "unit": "MW"},
    {"name": "actual_power_mw", "family": "base", "unit": "MW"},
    {"name": "delivery_ratio", "family": "base", "unit": "fraction"},
    {"name": "power_tracking_error_fraction", "family": "base", "unit": "fraction"},
    {"name": "requested_power_ramp_mw", "family": "base", "unit": "MW/interval"},
    {"name": "actual_power_ramp_mw", "family": "base", "unit": "MW/interval"},
    {"name": "soc", "family": "base", "unit": "fraction"},
    {"name": "soc_change", "family": "base", "unit": "fraction/interval"},
    {"name": "soc_volatility", "family": "base", "unit": "fraction"},
    {"name": "distance_to_soc_min", "family": "base", "unit": "fraction"},
    {"name": "distance_to_soc_max", "family": "base", "unit": "fraction"},
    {"name": "available_rack_fraction", "family": "base", "unit": "fraction"},
    {"name": "available_pcs_fraction", "family": "base", "unit": "fraction"},
    {"name": "available_power_fraction", "family": "base", "unit": "fraction"},
    {"name": "available_energy_fraction", "family": "base", "unit": "fraction"},
    {"name": "rte", "family": "base", "unit": "fraction"},
    {"name": "rte_trend", "family": "base", "unit": "fraction/hour"},
    {"name": "derating_frequency", "family": "base", "unit": "fraction"},
    {"name": "alarm_frequency", "family": "base", "unit": "fraction"},
    {"name": "recent_alarm_count", "family": "base", "unit": "count"},
    {"name": "operating_mode", "family": "base", "unit": "category"},
    {"name": "expected_actual_power_mw", "family": "residual", "unit": "MW"},
    {"name": "expected_power_residual_mw", "family": "residual", "unit": "MW"},
    {"name": "expected_absolute_power_residual_mw", "family": "residual", "unit": "MW"},
    {"name": "delivery_shortfall_mw", "family": "residual", "unit": "MW"},
    {"name": "expected_power_residual_rolling_mean", "family": "residual", "unit": "MW"},
    {"name": "delivery_shortfall_rolling_mean", "family": "residual", "unit": "MW"},
    {"name": "residual_persistence_count", "family": "residual", "unit": "count"},
    {"name": "site_max_thermal_residual_c", "family": "residual", "unit": "degC"},
    {"name": "site_mean_thermal_residual_c", "family": "residual", "unit": "degC"},
    {"name": "site_positive_thermal_residual_fraction", "family": "residual", "unit": "fraction"},
    {"name": "site_max_temperature_peer_z", "family": "peer", "unit": "dimensionless"},
    {"name": "site_mean_temperature_peer_deviation_c", "family": "peer", "unit": "degC"},
    {"name": "site_max_voltage_spread_peer_z", "family": "peer", "unit": "dimensionless"},
    {"name": "site_fraction_racks_unavailable", "family": "peer", "unit": "fraction"},
    {"name": "price_eur_per_mwh", "family": "market", "unit": "EUR/MWh"},
    {"name": "load_mw", "family": "market", "unit": "MW"},
    {"name": "renewable_generation_mw", "family": "market", "unit": "MW"},
    {"name": "residual_load_mw", "family": "market", "unit": "MW"},
    {"name": "renewable_share", "family": "market", "unit": "fraction"},
    {"name": "rolling_price_mean", "family": "market", "unit": "EUR/MWh"},
    {"name": "price_volatility", "family": "market", "unit": "EUR/MWh"},
)

_CATEGORICAL = ("operating_mode",)


def _is_leakage_name(name: str) -> bool:
    lowered = name.lower()
    return (
        is_prohibited_column(name)
        or lowered.startswith("future_")
        or lowered.startswith("failure_within_")
        or lowered in {"time_to_next_failure", "delivery_failure_at_timestamp"}
        or "future_failure" in lowered
    )


@dataclass(frozen=True)
class RiskDataset:
    keys: pd.DataFrame
    predictors: pd.DataFrame
    target: pd.Series
    evaluation_metadata: pd.DataFrame
    horizon_hours: int
    feature_manifest: tuple[dict[str, Any], ...]
    dataset_hash: str

    def subset(self, positions: NDArray[np.intp]) -> "RiskDataset":
        return RiskDataset(
            self.keys.iloc[positions].reset_index(drop=True),
            self.predictors.iloc[positions].reset_index(drop=True),
            self.target.iloc[positions].reset_index(drop=True),
            self.evaluation_metadata.iloc[positions].reset_index(drop=True),
            self.horizon_hours,
            self.feature_manifest,
            self.dataset_hash,
        )


@dataclass(frozen=True)
class PurgedRiskSplit:
    train: RiskDataset
    validation: RiskDataset
    calibration: RiskDataset
    test: RiskDataset


def _hash_dataset(keys: pd.DataFrame, predictors: pd.DataFrame, target: pd.Series) -> str:
    combined = pd.concat([keys, predictors, target.rename("__target__")], axis=1)
    schema = "|".join(f"{column}:{combined[column].dtype}" for column in combined)
    values = pd.util.hash_pandas_object(combined, index=False).to_numpy().tobytes()
    return hashlib.sha256(schema.encode() + values).hexdigest()


def build_risk_dataset(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    horizon_hours: int,
    *,
    families: tuple[str, ...] = ("base", "residual", "peer", "market"),
) -> RiskDataset:
    target_name = f"failure_within_{horizon_hours}h"
    leakage = [column for column in feature_frame if _is_leakage_name(str(column))]
    if leakage:
        raise ValueError(f"Leakage fields are prohibited from feature input: {leakage}")
    required_keys = ["timestamp_utc", "asset_id"]
    missing_keys = [column for column in required_keys if column not in feature_frame]
    if missing_keys or target_name not in target_frame:
        raise ValueError(f"Missing dataset keys or target: {missing_keys or [target_name]}")
    persisted_manifest = tuple(
        {
            **entry,
            "provenance": "DERIVED",
            "causal_status": "available_at_T",
            "included": entry["family"] in families and entry["name"] in feature_frame,
            "inclusion_reason": (
                "curated_causal_predictor"
                if entry["family"] in families and entry["name"] in feature_frame
                else "excluded_by_ablation_design"
                if entry["family"] not in families
                else "excluded_unavailable_in_source"
            ),
        }
        for entry in FEATURE_MANIFEST
    )
    selected = [entry["name"] for entry in persisted_manifest if entry["included"] is True]
    if not selected or not any(
        entry["family"] == "base" and entry["included"] is True for entry in persisted_manifest
    ):
        raise ValueError("No curated base operational predictors are available")
    keys = feature_frame[required_keys].copy()
    keys["timestamp_utc"] = pd.to_datetime(keys["timestamp_utc"], utc=True, errors="raise")
    if keys.duplicated(required_keys).any():
        raise ValueError("Risk features must be unique by timestamp_utc and asset_id")
    targets = target_frame[[*required_keys, target_name]].copy()
    targets["timestamp_utc"] = pd.to_datetime(targets["timestamp_utc"], utc=True, errors="raise")
    joined = keys.merge(targets, on=required_keys, how="left", validate="one_to_one")
    order = np.argsort(keys["timestamp_utc"].to_numpy(), kind="stable")
    keys = keys.iloc[order].reset_index(drop=True)
    predictors = feature_frame[selected].iloc[order].reset_index(drop=True)
    target = joined[target_name].iloc[order].astype("Float64").reset_index(drop=True)
    metadata_columns = [
        column
        for column in [
            "fault_id",
            "fault_type",
            "fault_severity",
            "failure_event_id",
            f"future_failure_event_ids_{horizon_hours}h",
        ]
        if column in target_frame
    ]
    if metadata_columns:
        meta_source = target_frame[[*required_keys, *metadata_columns]]
        metadata = keys.merge(meta_source, on=required_keys, how="left", validate="one_to_one")
        metadata = metadata[metadata_columns]
    else:
        metadata = pd.DataFrame(index=keys.index)
    return RiskDataset(
        keys,
        predictors,
        target,
        metadata.reset_index(drop=True),
        horizon_hours,
        persisted_manifest,
        _hash_dataset(keys, predictors, target),
    )


def purged_chronological_split(dataset: RiskDataset, config: DeliveryRiskConfig) -> PurgedRiskSplit:
    available = np.flatnonzero(dataset.target.notna().to_numpy())
    clean = dataset.subset(available)
    timestamps = pd.DatetimeIndex(clean.keys["timestamp_utc"].unique()).sort_values()
    if len(timestamps) < 8:
        raise ValueError("Too few uncensored timestamps for four chronological partitions")
    cuts = [
        int(len(timestamps) * config.train_fraction),
        int(len(timestamps) * (config.train_fraction + config.validation_fraction)),
        int(
            len(timestamps)
            * (config.train_fraction + config.validation_fraction + config.calibration_fraction)
        ),
    ]
    cuts = [max(1, min(value, len(timestamps) - 1)) for value in cuts]
    if not cuts[0] < cuts[1] < cuts[2]:
        raise ValueError("Chronological split boundaries collapsed")
    boundaries = [timestamps[index] for index in cuts]
    if "fault_id" in clean.evaluation_metadata:
        event_frame = pd.DataFrame(
            {
                "timestamp_utc": clean.keys["timestamp_utc"],
                "fault_id": clean.evaluation_metadata["fault_id"],
            }
        ).dropna(subset=["fault_id"])
        protected: list[pd.Timestamp] = []
        for boundary in boundaries:
            adjusted = boundary
            for _, event in event_frame.groupby("fault_id"):
                event_start = event["timestamp_utc"].min()
                event_end = event["timestamp_utc"].max()
                if event_start < boundary <= event_end:
                    adjusted = min(adjusted, event_start)
            protected.append(adjusted)
        boundaries = protected
        if not boundaries[0] < boundaries[1] < boundaries[2]:
            raise ValueError("Event protection collapsed chronological split boundaries")
    horizon = pd.Timedelta(hours=dataset.horizon_hours)
    time = clean.keys["timestamp_utc"]
    masks = [
        (time < boundaries[0]) & (time + horizon < boundaries[0]),
        (time >= boundaries[0]) & (time < boundaries[1]) & (time + horizon < boundaries[1]),
        (time >= boundaries[1]) & (time < boundaries[2]) & (time + horizon < boundaries[2]),
        time >= boundaries[2],
    ]
    partitions = [clean.subset(np.flatnonzero(mask.to_numpy())) for mask in masks]
    if any(partition.keys.empty for partition in partitions):
        raise ValueError("Horizon purge produced an empty partition; generate more history")
    split = PurgedRiskSplit(*partitions)
    ordered = [split.train, split.validation, split.calibration, split.test]
    if any(
        left.keys["timestamp_utc"].max() >= right.keys["timestamp_utc"].min()
        for left, right in pairwise(ordered)
    ):
        raise AssertionError("Purged partitions are not strictly chronological")
    if any(
        left.keys["timestamp_utc"].max() + horizon >= right.keys["timestamp_utc"].min()
        for left, right in pairwise(ordered)
    ):
        raise AssertionError("A target horizon crosses a partition boundary")
    return split


def numeric_and_categorical(dataset: RiskDataset) -> tuple[list[str], list[str]]:
    categorical = [column for column in _CATEGORICAL if column in dataset.predictors]
    numeric = [str(column) for column in dataset.predictors if column not in categorical]
    return numeric, categorical


def dataset_distribution(dataset: RiskDataset) -> dict[str, Any]:
    available = dataset.target.notna()
    target = dataset.target.loc[available].astype(int)
    return {
        "horizon_hours": dataset.horizon_hours,
        "eligible_rows": int(available.sum()),
        "censored_rows": int((~available).sum()),
        "positive_rows": int(target.sum()),
        "negative_rows": int((target == 0).sum()),
        "positive_prevalence": float(target.mean()) if len(target) else float("nan"),
        "time_range": [
            str(dataset.keys["timestamp_utc"].min()),
            str(dataset.keys["timestamp_utc"].max()),
        ],
    }
