"""Time-weighted availability KPIs, downtime, and deterministic events."""

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def interval_weights_hours(timestamps: pd.Series) -> pd.Series:
    ordered = pd.to_datetime(timestamps, utc=True, errors="raise")
    if ordered.duplicated().any() or not ordered.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and chronological")
    differences = ordered.shift(-1) - ordered
    positive = differences.loc[differences.gt(pd.Timedelta(0))]
    fallback = positive.median() if len(positive) else pd.Timedelta(0)
    differences.iloc[-1] = fallback
    return differences.dt.total_seconds().div(3600).fillna(0.0)


def time_weighted_mean(values: pd.Series, timestamps: pd.Series) -> float:
    weights = interval_weights_hours(timestamps)
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid]))


def availability_summary(snapshots: pd.DataFrame, *, tolerance: float = 1e-6) -> dict[str, Any]:
    ordered = snapshots.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    weights = interval_weights_hours(ordered["timestamp_utc"])
    active = ordered["requested_power_is_active"].astype(bool)
    supported = ordered["requested_power_availability"].ge(1 - tolerance) & active
    observed_success = ordered["observed_delivery_success"].eq(True) & active
    return {
        "start_timestamp": ordered["timestamp_utc"].min(),
        "end_timestamp": ordered["timestamp_utc"].max(),
        "snapshot_count": len(ordered),
        "mean_technical_availability": time_weighted_mean(
            ordered["technical_availability"], ordered["timestamp_utc"]
        ),
        "minimum_technical_availability": float(ordered["technical_availability"].min()),
        "mean_discharge_power_availability": time_weighted_mean(
            ordered["discharge_power_availability"], ordered["timestamp_utc"]
        ),
        "minimum_available_discharge_power_mw": float(
            ordered["available_discharge_power_mw"].min()
        ),
        "mean_discharge_energy_availability": time_weighted_mean(
            ordered["discharge_energy_availability"], ordered["timestamp_utc"]
        ),
        "minimum_available_discharge_energy_mwh": float(
            ordered["available_discharge_energy_mwh"].min()
        ),
        "active_requested_power_intervals": int(active.sum()),
        "request_support_rate": float(supported.sum() / active.sum()) if active.any() else None,
        "observed_delivery_success_rate": (
            float(observed_success.sum() / active.sum()) if active.any() else None
        ),
        "mean_requested_power_availability": time_weighted_mean(
            ordered["requested_power_availability"], ordered["timestamp_utc"]
        ),
        "hours_technical_availability_below_one": float(
            weights.loc[ordered["technical_availability"].lt(1 - tolerance)].sum()
        ),
    }


def component_downtime(
    capabilities: pd.DataFrame, *, id_column: str = "component_id"
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, frame in capabilities.groupby(id_column, sort=True):
        ordered = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
        weights = interval_weights_hours(ordered["timestamp_utc"]) * 60
        unavailable = ordered["state"].astype(str).eq("UNAVAILABLE")
        known = ~ordered["state"].astype(str).eq("UNKNOWN")
        starts = unavailable & ~unavailable.shift(fill_value=False)
        groups = (~unavailable).cumsum()
        longest = float(weights.where(unavailable, 0).groupby(groups).sum().max())
        known_minutes = float(weights.loc[known].sum())
        available_minutes = float(weights.loc[ordered["state"].astype(str).eq("AVAILABLE")].sum())
        rows.append(
            {
                "component_id": component,
                "downtime_minutes": float(weights.loc[unavailable].sum()),
                "availability_fraction": (
                    available_minutes / known_minutes if known_minutes > 0 else None
                ),
                "number_of_outage_events": int(starts.sum()),
                "longest_outage_minutes": longest,
                "unknown_minutes": float(weights.loc[~known].sum()),
                "data_provenance": "DERIVED ENGINEERING ANALYTIC",
            }
        )
    return pd.DataFrame(rows)


def create_availability_events(
    capabilities: pd.DataFrame, *, tolerance: float = 1e-6
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, frame in capabilities.groupby("component_id", sort=True):
        ordered = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
        weights = interval_weights_hours(ordered["timestamp_utc"])
        nominal = ordered.get("nominal_power_mw", pd.Series(np.nan, index=ordered.index))
        fraction = ordered["available_discharge_power_mw"].div(nominal).clip(0, 1)
        event_type = pd.Series("NONE", index=ordered.index)
        event_type.loc[ordered["state"].astype(str).eq("UNAVAILABLE")] = "DOWNTIME"
        event_type.loc[
            ordered["state"].astype(str).eq("AVAILABLE") & fraction.lt(1 - tolerance)
        ] = "DERATED"
        active = event_type.ne("NONE")
        groups = ((event_type != event_type.shift()) | ~active).cumsum()
        for _, event in ordered.loc[active].groupby(groups.loc[active], sort=False):
            index = event.index
            kind = str(event_type.loc[index[0]])
            start = event["timestamp_utc"].iloc[0]
            duration = float(weights.loc[index].sum() * 60)
            end = pd.Timestamp(start) + pd.Timedelta(minutes=duration)
            raw_id = f"{component}|{kind}|{pd.Timestamp(start).isoformat()}"
            rows.append(
                {
                    "event_id": "AVL-" + hashlib.sha256(raw_id.encode()).hexdigest()[:16],
                    "component_id": component,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "event_type": kind,
                    "minimum_available_fraction": float(fraction.loc[index].min()),
                    "duration_minutes": duration,
                    "data_provenance": "DERIVED ENGINEERING ANALYTIC",
                }
            )
    return pd.DataFrame(rows)
