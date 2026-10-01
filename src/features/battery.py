"""Causal site/rack operational features and transparent KPI summaries."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from besspulse.config import SimulationConfig
from features.validation import (
    exact_time_lag,
    prepare_temporal_frame,
    trailing_statistic,
    validate_feature_frame,
)


def _rename_available(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return frame.rename(columns={key: value for key, value in mapping.items() if key in frame})


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing telemetry columns: {missing}")


def _per_entity(
    frame: pd.DataFrame,
    entity: str,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    pieces = [transform(group.copy()) for _, group in frame.groupby(entity, sort=False)]
    return pd.concat(pieces, ignore_index=True) if pieces else frame.copy()


def _time_since_event_minutes(timestamps: pd.Series, event: pd.Series) -> pd.Series:
    event_times = timestamps.where(event.astype(bool)).ffill()
    result = (timestamps - event_times).dt.total_seconds() / 60
    return result.where(event_times.notna(), np.nan)


def build_site_features(
    telemetry: pd.DataFrame,
    config: SimulationConfig | None = None,
    *,
    asset_id: str = "BESS-001",
) -> pd.DataFrame:
    """Create one causal site feature row per `(timestamp_utc, asset_id)`."""

    cfg = config or SimulationConfig()
    frame = _rename_available(
        telemetry,
        {
            "site_requested_power_mw": "requested_power_mw",
            "site_actual_power_mw": "actual_power_mw",
            "site_soc": "soc",
        },
    )
    if "asset_id" not in frame:
        frame["asset_id"] = asset_id
    required = [
        "timestamp_utc",
        "asset_id",
        "requested_power_mw",
        "actual_power_mw",
        "available_power_mw",
        "available_energy_mwh",
        "soc",
        "rte",
        "available_racks",
        "available_pcs",
        "operating_mode",
        "alarm_count",
    ]
    optional = [column for column in ["ambient_temperature_c"] if column in frame]
    _require(frame, required)
    selected = frame[[*required, *optional]].copy()
    selected = prepare_temporal_frame(selected, ["asset_id"])
    battery = cfg.battery
    feature_cfg = cfg.features
    selected["rated_power_mw"] = battery.site_rated_power_mw
    selected["rated_energy_mwh"] = battery.site_rated_energy_mwh
    selected["active_power_interval"] = (
        selected["requested_power_mw"].abs() > cfg.minimum_request_threshold_mw
    )
    selected["delivery_ratio"] = np.where(
        selected["active_power_interval"],
        selected["actual_power_mw"].abs() / selected["requested_power_mw"].abs(),
        np.nan,
    )
    selected["power_residual_mw"] = selected["actual_power_mw"] - selected["requested_power_mw"]
    selected["absolute_power_residual_mw"] = selected["power_residual_mw"].abs()
    selected["normalized_power_residual"] = (
        selected["power_residual_mw"] / battery.site_rated_power_mw
    )
    selected["power_tracking_error_fraction"] = np.where(
        selected["active_power_interval"],
        selected["absolute_power_residual_mw"] / selected["requested_power_mw"].abs(),
        np.nan,
    )

    def add_entity_features(group: pd.DataFrame) -> pd.DataFrame:
        timestamps: pd.Series[pd.Timestamp] = group["timestamp_utc"]
        residual = group["power_residual_mw"]
        absolute = group["absolute_power_residual_mw"]
        window = feature_cfg.power_residual_window
        group["rolling_power_residual_mean"] = trailing_statistic(
            residual,
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["rolling_power_residual_std"] = trailing_statistic(
            residual,
            timestamps,
            window,
            "std",
            min_periods=2,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["rolling_absolute_power_residual"] = trailing_statistic(
            absolute,
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["requested_power_ramp_mw"] = group["requested_power_mw"].diff()
        group["actual_power_ramp_mw"] = group["actual_power_mw"].diff()
        group["soc_change"] = group["soc"].diff()
        elapsed_h = timestamps.diff().dt.total_seconds() / 3600
        group["soc_change_rate"] = group["soc_change"] / elapsed_h.replace(0, np.nan)
        group["soc_volatility"] = trailing_statistic(
            group["soc"],
            timestamps,
            feature_cfg.soc_volatility_window,
            "std",
            min_periods=2,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_high_soc"] = trailing_statistic(
            (group["soc"] >= feature_cfg.high_soc_threshold).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_low_soc"] = trailing_statistic(
            (group["soc"] <= feature_cfg.low_soc_threshold).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["rte_change"] = group["rte"].diff()
        lagged_rte = exact_time_lag(group["rte"], timestamps, feature_cfg.rte_trend_window)
        trend_hours = pd.Timedelta(feature_cfg.rte_trend_window).total_seconds() / 3600
        group["rte_trend"] = (group["rte"] - lagged_rte) / trend_hours
        group["alarm_active"] = group["alarm_count"] > 0
        group["recent_alarm_count"] = trailing_statistic(
            group["alarm_count"].astype(float),
            timestamps,
            feature_cfg.alarm_window,
            "sum",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["alarm_frequency"] = trailing_statistic(
            group["alarm_active"].astype(float),
            timestamps,
            feature_cfg.alarm_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_since_last_alarm_minutes"] = _time_since_event_minutes(
            timestamps, group["alarm_active"]
        )
        group["derating_frequency"] = trailing_statistic(
            group["derating_indicator"].astype(float),
            timestamps,
            feature_cfg.availability_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        return group

    selected["distance_to_soc_min"] = selected["soc"] - battery.soc_min
    selected["distance_to_soc_max"] = battery.soc_max - selected["soc"]
    selected["available_rack_fraction"] = selected["available_racks"] / (
        battery.pcs_count * battery.racks_per_pcs
    )
    selected["available_pcs_fraction"] = selected["available_pcs"] / battery.pcs_count
    selected["available_power_fraction"] = (
        selected["available_power_mw"] / battery.site_rated_power_mw
    )
    selected["available_energy_fraction"] = (
        selected["available_energy_mwh"] / battery.site_rated_energy_mwh
    )
    selected["technical_availability"] = selected[
        ["available_rack_fraction", "available_pcs_fraction"]
    ].min(axis=1)
    selected["derating_indicator"] = selected["available_power_fraction"] < (1 - 1e-9)
    mode = selected["operating_mode"].astype(str).str.upper()
    selected["is_charging"] = mode == "CHARGING"
    selected["is_discharging"] = mode == "DISCHARGING"
    selected["is_idle"] = mode == "IDLE"
    result = _per_entity(selected, "asset_id", add_entity_features)
    result.attrs = {
        "data_provenance": "DERIVED",
        "entity_level": "site",
        "power_sign_convention": "positive_discharge_negative_charge",
    }
    validate_feature_frame(result, ["asset_id"])
    return result


def build_rack_features(
    telemetry: pd.DataFrame,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Create causal operational features keyed by `(timestamp_utc, rack_id)`."""

    cfg = config or SimulationConfig()
    feature_cfg = cfg.features
    required = [
        "timestamp_utc",
        "rack_id",
        "pcs_id",
        "soc",
        "soh_proxy",
        "voltage_v",
        "current_a",
        "temperature_mean_c",
        "temperature_spread_c",
        "voltage_spread_v",
        "requested_power_mw",
        "actual_power_mw",
        "cumulative_throughput_mwh",
        "equivalent_full_cycles",
        "rte",
        "availability",
        "alarm_code",
        "operating_state",
    ]
    _require(telemetry, required)
    optional = [column for column in ["ambient_temperature_c"] if column in telemetry]
    frame = prepare_temporal_frame(telemetry[[*required, *optional]].copy(), ["rack_id"])
    frame["rack_power_residual_mw"] = frame["actual_power_mw"] - frame["requested_power_mw"]
    frame["absolute_rack_power_residual_mw"] = frame["rack_power_residual_mw"].abs()
    frame["distance_to_soc_min"] = frame["soc"] - cfg.battery.soc_min
    frame["distance_to_soc_max"] = cfg.battery.soc_max - frame["soc"]
    frame["alarm_indicator"] = frame["alarm_code"].notna() & frame["alarm_code"].ne("")
    if "ambient_temperature_c" in frame:
        frame["temperature_vs_ambient_c"] = (
            frame["temperature_mean_c"] - frame["ambient_temperature_c"]
        )
    else:
        frame["temperature_vs_ambient_c"] = np.nan
    mode = frame["operating_state"].astype(str).str.upper()
    frame["is_charging"] = mode == "CHARGING"
    frame["is_discharging"] = mode == "DISCHARGING"
    frame["is_idle"] = mode == "IDLE"

    def add_entity_features(group: pd.DataFrame) -> pd.DataFrame:
        timestamps: pd.Series[pd.Timestamp] = group["timestamp_utc"]
        elapsed_h = timestamps.diff().dt.total_seconds() / 3600
        group["soc_change"] = group["soc"].diff()
        group["soc_change_rate"] = group["soc_change"] / elapsed_h.replace(0, np.nan)
        group["soc_volatility"] = trailing_statistic(
            group["soc"],
            timestamps,
            feature_cfg.soc_volatility_window,
            "std",
            min_periods=2,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_high_soc"] = trailing_statistic(
            (group["soc"] >= feature_cfg.high_soc_threshold).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_low_soc"] = trailing_statistic(
            (group["soc"] <= feature_cfg.low_soc_threshold).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["recent_charge_fraction"] = trailing_statistic(
            (group["actual_power_mw"] < 0).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["recent_discharge_fraction"] = trailing_statistic(
            (group["actual_power_mw"] > 0).astype(float),
            timestamps,
            feature_cfg.soc_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["temperature_ramp_c"] = group["temperature_mean_c"].diff()
        group["rolling_temperature_mean"] = trailing_statistic(
            group["temperature_mean_c"],
            timestamps,
            feature_cfg.temperature_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["rolling_temperature_std"] = trailing_statistic(
            group["temperature_mean_c"],
            timestamps,
            feature_cfg.temperature_window,
            "std",
            min_periods=2,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["thermal_exposure_above_threshold"] = trailing_statistic(
            (group["temperature_mean_c"] > feature_cfg.thermal_threshold_c).astype(float),
            timestamps,
            feature_cfg.thermal_exposure_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["voltage_spread_change"] = group["voltage_spread_v"].diff()
        group["current_change"] = group["current_a"].diff()
        group["rack_power_change"] = group["actual_power_mw"].diff()
        group["rte_change"] = group["rte"].diff()
        lagged_rte = exact_time_lag(group["rte"], timestamps, feature_cfg.rte_trend_window)
        trend_hours = pd.Timedelta(feature_cfg.rte_trend_window).total_seconds() / 3600
        group["rte_trend"] = (group["rte"] - lagged_rte) / trend_hours
        group["availability_rate"] = trailing_statistic(
            group["availability"].astype(float),
            timestamps,
            feature_cfg.availability_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["alarm_frequency"] = trailing_statistic(
            group["alarm_indicator"].astype(float),
            timestamps,
            feature_cfg.alarm_window,
            "mean",
            min_periods=1,
            require_full_window=feature_cfg.require_full_windows,
        )
        group["time_since_last_alarm_minutes"] = _time_since_event_minutes(
            timestamps, group["alarm_indicator"]
        )
        group["throughput_change_mwh"] = group["cumulative_throughput_mwh"].diff()
        group["temperature_exposure"] = group["thermal_exposure_above_threshold"]
        group["high_soc_exposure"] = group["time_high_soc"]
        group["low_soc_exposure"] = group["time_low_soc"]
        return group

    result = _per_entity(frame, "rack_id", add_entity_features)
    result.attrs = {
        "data_provenance": "DERIVED",
        "entity_level": "rack",
        "power_sign_convention": "positive_discharge_negative_charge",
    }
    validate_feature_frame(result, ["rack_id"])
    return result


def summarize_site_kpis(
    site_features: pd.DataFrame,
    config: SimulationConfig | None = None,
    rack_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize observed interval performance; this is not a prediction."""

    if site_features.empty:
        raise ValueError("Cannot summarize an empty site feature frame")
    cfg = config or SimulationConfig()
    ordered = prepare_temporal_frame(site_features, ["asset_id"])
    latest = ordered.iloc[-1]
    active = ordered.loc[ordered["active_power_interval"]]
    throughput = float("nan")
    if rack_features is not None and not rack_features.empty:
        changes = rack_features["throughput_change_mwh"].clip(lower=0)
        throughput = float(changes.sum(skipna=True))
    delivered = active["delivery_ratio"] >= cfg.delivery_failure_ratio
    return {
        "rated_power_mw": cfg.battery.site_rated_power_mw,
        "rated_energy_mwh": cfg.battery.site_rated_energy_mwh,
        "available_power_mw": float(latest["available_power_mw"]),
        "available_energy_mwh": float(latest["available_energy_mwh"]),
        "site_soc": float(latest["soc"]),
        "rte": float(latest["rte"]),
        "technical_availability": float(ordered["technical_availability"].mean()),
        "requested_power_availability": (
            float(delivered.mean()) if not delivered.empty else float("nan")
        ),
        "available_rack_fraction": float(ordered["available_rack_fraction"].mean()),
        "available_pcs_fraction": float(ordered["available_pcs_fraction"].mean()),
        "delivery_ratio_mean": float(active["delivery_ratio"].mean()),
        "alarm_count": int(ordered["alarm_count"].sum()),
        "energy_throughput_mwh": throughput,
        "data_provenance": "DERIVED",
    }
