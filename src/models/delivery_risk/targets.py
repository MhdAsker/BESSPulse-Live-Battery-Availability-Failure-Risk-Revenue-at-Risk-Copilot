"""Future delivery-failure targets, isolated from causal feature generation."""

import numpy as np
import pandas as pd

from models.delivery_risk.config import DeliveryRiskConfig


def _validate_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp_utc", "asset_id", "requested_power_mw", "actual_power_mw"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing target-source columns: {missing}")
    result = frame[required].copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True, errors="raise")
    if result.duplicated(["timestamp_utc", "asset_id"]).any():
        raise ValueError("Target source must be unique by timestamp_utc and asset_id")
    for column in ["requested_power_mw", "actual_power_mw"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if np.isinf(result[["requested_power_mw", "actual_power_mw"]].to_numpy()).any():
        raise ValueError("Target source contains infinite power values")
    return result.sort_values(["asset_id", "timestamp_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _label_entity(group: pd.DataFrame, config: DeliveryRiskConfig) -> pd.DataFrame:
    result = group.copy()
    active = result["requested_power_mw"].abs().gt(config.minimum_request_threshold_mw)
    delivery_ratio = result["actual_power_mw"].abs().div(result["requested_power_mw"].abs())
    result["target_interval_eligible"] = active
    result["target_delivery_ratio"] = delivery_ratio.where(active)
    failed = active & delivery_ratio.lt(config.failure_delivery_ratio_threshold)
    result["delivery_failure_at_timestamp"] = failed
    starts = failed & ~failed.shift(fill_value=False)
    event_number = starts.cumsum()
    event_ids = pd.Series(
        [
            f"{asset_id}-FAIL-{number:04d}" if is_failed else None
            for asset_id, number, is_failed in zip(
                result["asset_id"], event_number, failed, strict=True
            )
        ],
        index=result.index,
        dtype="string",
    )
    result["failure_event_id"] = event_ids.astype("string")

    timestamps = result["timestamp_utc"].astype("int64").to_numpy()
    failure_times = timestamps[failed.to_numpy()]
    failure_event_ids = event_ids.loc[failed].to_numpy(dtype=str)
    for horizon in config.horizons_hours:
        horizon_ns = int(pd.Timedelta(hours=horizon).value)
        left = np.searchsorted(failure_times, timestamps, side="right")
        right = np.searchsorted(failure_times, timestamps + horizon_ns, side="right")
        labels = (right > left).astype(float)
        fully_observed = timestamps + horizon_ns <= timestamps[-1]
        labels[~fully_observed] = np.nan
        result[f"failure_within_{horizon}h"] = labels
        result[f"target_available_{horizon}h"] = fully_observed
        result[f"future_failure_event_ids_{horizon}h"] = [
            tuple(dict.fromkeys(failure_event_ids[start:end])) if observed else tuple()
            for start, end, observed in zip(left, right, fully_observed, strict=True)
        ]
    return result


def generate_delivery_failure_targets(
    telemetry: pd.DataFrame, config: DeliveryRiskConfig | None = None
) -> pd.DataFrame:
    """Label failures in `(T, T+h]`; rows lacking the full horizon are censored."""

    cfg = config or DeliveryRiskConfig()
    source = _validate_telemetry(telemetry)
    pieces = [_label_entity(group, cfg) for _, group in source.groupby("asset_id", sort=False)]
    output = pd.concat(pieces, ignore_index=True)
    output.attrs = {
        "data_provenance": "DERIVED FROM SIMULATED TELEMETRY",
        "horizon_semantics": "(T, T+h]",
        "failure_delivery_ratio_threshold": cfg.failure_delivery_ratio_threshold,
        "minimum_request_threshold_mw": cfg.minimum_request_threshold_mw,
    }
    return output


def assign_failure_event_ids(targets: pd.DataFrame) -> pd.Series:
    """Identify contiguous observed failure episodes for independent-event reporting."""

    output = pd.Series(pd.NA, index=targets.index, dtype="string")
    for asset_id, group in targets.groupby("asset_id", sort=False):
        failed = group["delivery_failure_at_timestamp"].astype(bool)
        starts = failed & ~failed.shift(fill_value=False)
        event_number = starts.cumsum()
        labels = pd.Series(
            [f"{asset_id}-FAIL-{number:04d}" for number in event_number],
            index=group.index,
            dtype="string",
        )
        output.loc[group.index[failed]] = labels.loc[failed]
    return output
