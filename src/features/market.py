"""Causal transformations of normalized REAL ENTSO-E observations."""

import numpy as np
import pandas as pd

from besspulse.config import SimulationConfig
from data.parser import parse_duration
from features.validation import (
    exact_time_lag,
    prepare_temporal_frame,
    trailing_statistic,
    validate_feature_frame,
)

_METRIC_COLUMNS = {
    "day_ahead_price": "price_eur_per_mwh",
    "actual_load": "load_mw",
    "load_forecast": "load_forecast_mw",
    "wind_onshore_generation": "wind_onshore_generation_mw",
    "wind_offshore_generation": "wind_offshore_generation_mw",
    "solar_generation": "solar_generation_mw",
    "wind_onshore_generation_forecast": "wind_onshore_forecast_mw",
    "wind_offshore_generation_forecast": "wind_offshore_forecast_mw",
    "solar_generation_forecast": "solar_forecast_mw",
}


def _strict_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return a sum only when every required component is observed."""

    values = frame.reindex(columns=columns)
    return values.sum(axis=1, min_count=len(columns))


def _resolution_minutes(value: object) -> float:
    if not isinstance(value, str):
        return float("nan")
    return parse_duration(value).total_seconds() / 60


def build_market_features(
    observations: pd.DataFrame,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Pivot normalized ENTSO-E rows and calculate causal market features."""

    cfg = config or SimulationConfig()
    required = ["timestamp_utc", "metric", "value"]
    missing = [column for column in required if column not in observations]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")
    selected_columns = [
        column
        for column in ["timestamp_utc", "metric", "value", "resolution"]
        if column in observations
    ]
    long = observations[selected_columns].copy()
    long["metric"] = long["metric"].map(
        lambda value: value.value if hasattr(value, "value") else str(value)
    )
    long = long.loc[long["metric"].isin(_METRIC_COLUMNS)].copy()
    if long.empty:
        raise ValueError("No supported market metrics are present")
    long = prepare_temporal_frame(long, ["metric"])
    wide = (
        long.pivot(index="timestamp_utc", columns="metric", values="value")
        .rename(columns=_METRIC_COLUMNS)
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    if "resolution" in long:
        resolution = (
            long.assign(resolution_minutes=long["resolution"].map(_resolution_minutes))
            .groupby("timestamp_utc", as_index=False)["resolution_minutes"]
            .min()
        )
        wide = wide.merge(resolution, on="timestamp_utc", how="left", validate="one_to_one")
        wide = wide.rename(columns={"resolution_minutes": "market_source_resolution_minutes"})
    else:
        wide["market_source_resolution_minutes"] = np.nan

    for column in _METRIC_COLUMNS.values():
        if column not in wide:
            wide[column] = np.nan
    wide["wind_generation_mw"] = _strict_sum(
        wide, ["wind_onshore_generation_mw", "wind_offshore_generation_mw"]
    )
    wide["renewable_generation_mw"] = _strict_sum(
        wide, ["wind_generation_mw", "solar_generation_mw"]
    )
    wide["residual_load_mw"] = wide["load_mw"] - wide["renewable_generation_mw"]
    wide["renewable_share"] = np.where(
        wide["load_mw"].notna() & wide["load_mw"].ne(0),
        wide["renewable_generation_mw"] / wide["load_mw"],
        np.nan,
    )
    wide["wind_forecast_mw"] = _strict_sum(
        wide, ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw"]
    )
    wide["renewable_forecast_mw"] = _strict_sum(wide, ["wind_forecast_mw", "solar_forecast_mw"])

    timestamps = wide["timestamp_utc"]
    price = wide["price_eur_per_mwh"]
    wide["price_lag_24h"] = exact_time_lag(price, timestamps, "24h")
    wide["price_lag_168h"] = exact_time_lag(price, timestamps, "168h")
    wide["rolling_price_mean"] = trailing_statistic(
        price,
        timestamps,
        cfg.features.market_price_window,
        "mean",
        min_periods=1,
        require_full_window=cfg.features.require_full_windows,
    )
    wide["price_volatility"] = trailing_statistic(
        price,
        timestamps,
        cfg.features.market_volatility_window,
        "std",
        min_periods=2,
        require_full_window=cfg.features.require_full_windows,
    )
    wide["price_change_1h"] = price - exact_time_lag(price, timestamps, "1h")
    wide["price_change_24h"] = price - wide["price_lag_24h"]
    local = timestamps.dt.tz_convert(cfg.market.market_timezone)
    wide["hour_of_day"] = local.dt.hour
    wide["day_of_week"] = local.dt.dayofweek
    wide["hour_of_week"] = wide["day_of_week"] * 24 + wide["hour_of_day"]
    wide["is_weekend"] = wide["day_of_week"] >= 5
    wide["data_provenance"] = "DERIVED"
    wide.attrs = {
        "data_provenance": "DERIVED",
        "source": "ENTSO-E Transparency Platform",
        "column_provenance": {name: "REAL" for name in _METRIC_COLUMNS.values()},
    }
    validate_feature_frame(wide, [])
    return wide


def align_market_to_battery(
    battery_features: pd.DataFrame,
    market_features: pd.DataFrame,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Backward as-of join with a finite age tolerance; future market rows are forbidden."""

    cfg = config or SimulationConfig()
    if "asset_id" not in battery_features:
        raise ValueError("Battery feature frame requires asset_id")
    battery = prepare_temporal_frame(battery_features, ["asset_id"])
    market = prepare_temporal_frame(market_features, [])
    market = market.rename(columns={"timestamp_utc": "market_timestamp_utc"})
    left = battery.sort_values("timestamp_utc", kind="mergesort")
    right = market.sort_values("market_timestamp_utc", kind="mergesort")
    result = pd.merge_asof(
        left,
        right,
        left_on="timestamp_utc",
        right_on="market_timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta(cfg.features.market_alignment_tolerance),
        allow_exact_matches=True,
        suffixes=("", "_market"),
    )
    future = result["market_timestamp_utc"] > result["timestamp_utc"]
    if future.fillna(False).any():
        raise AssertionError("Future market observation entered backward join")
    result["market_data_age_minutes"] = (
        result["timestamp_utc"] - result["market_timestamp_utc"]
    ).dt.total_seconds() / 60
    source_expired = (
        result["market_source_resolution_minutes"].notna()
        & (result["market_data_age_minutes"] >= result["market_source_resolution_minutes"])
        if "market_source_resolution_minutes" in result
        else pd.Series(False, index=result.index)
    )
    if source_expired.any():
        right_payload = [
            (f"{column}_market" if column in left.columns else column)
            for column in right.columns
            if column != "market_timestamp_utc"
        ]
        for column in right_payload:
            result.loc[source_expired, column] = np.nan
        result.loc[source_expired, "market_timestamp_utc"] = pd.NaT
        result.loc[source_expired, "market_data_age_minutes"] = np.nan
    result["market_data_available"] = result["market_timestamp_utc"].notna()
    result["data_provenance"] = "DERIVED"
    result = result.sort_values(["asset_id", "timestamp_utc"], kind="mergesort").reset_index(
        drop=True
    )
    result.attrs = {
        "data_provenance": "DERIVED",
        "join_direction": "backward",
        "join_tolerance": cfg.features.market_alignment_tolerance,
    }
    validate_feature_frame(result, ["asset_id"])
    return result
