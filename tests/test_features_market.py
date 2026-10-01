from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from besspulse.config import FeatureConfig, SimulationConfig
from features.market import align_market_to_battery, build_market_features


def market_config(**updates) -> SimulationConfig:
    values = {
        "market_price_window": "24h",
        "market_volatility_window": "24h",
        "market_alignment_tolerance": "10min",
        "require_full_windows": False,
    }
    values.update(updates)
    return SimulationConfig(features=FeatureConfig(**values))


def price_observations(periods: int = 169) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="1h", tz=UTC)
    values = np.arange(periods, dtype=float)
    if periods > 5:
        values[5] = -10.0
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "metric": "day_ahead_price",
            "value": values,
            "resolution": "PT60M",
        }
    )


def test_price_lags_use_exact_timestamps_and_preserve_negative_prices() -> None:
    result = build_market_features(price_observations(), market_config())
    assert result.loc[5, "price_eur_per_mwh"] == -10.0
    assert result.loc[24, "price_lag_24h"] == 0.0
    assert result.loc[168, "price_lag_168h"] == 0.0
    assert result.loc[24, "price_change_24h"] == 24.0


def test_price_rolling_mean_and_volatility_are_causal() -> None:
    source = price_observations(30)
    baseline = build_market_features(source, market_config())
    changed = source.copy()
    changed.loc[29, "value"] = 1_000_000.0
    recomputed = build_market_features(changed, market_config())
    pd.testing.assert_series_equal(
        baseline.loc[:28, "rolling_price_mean"],
        recomputed.loc[:28, "rolling_price_mean"],
    )
    pd.testing.assert_series_equal(
        baseline.loc[:28, "price_volatility"],
        recomputed.loc[:28, "price_volatility"],
    )


def test_generation_aggregates_residual_load_and_share_require_all_components() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz=UTC)
    rows = pd.DataFrame(
        {
            "timestamp_utc": [timestamp] * 4,
            "metric": [
                "actual_load",
                "wind_onshore_generation",
                "wind_offshore_generation",
                "solar_generation",
            ],
            "value": [100.0, 30.0, 20.0, 10.0],
            "resolution": ["PT60M"] * 4,
        }
    )
    result = build_market_features(rows, market_config()).iloc[0]
    assert result["wind_generation_mw"] == 50.0
    assert result["renewable_generation_mw"] == 60.0
    assert result["residual_load_mw"] == 40.0
    assert result["renewable_share"] == pytest.approx(0.6)
    missing = build_market_features(rows.iloc[[0, 1, 3]], market_config()).iloc[0]
    assert pd.isna(missing["wind_generation_mw"])
    assert pd.isna(missing["renewable_generation_mw"])
    assert pd.isna(missing["residual_load_mw"])


def test_strict_24h_lag_remains_utc_correct_across_spring_dst() -> None:
    rows = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2026-03-28T12:00:00Z"),
                pd.Timestamp("2026-03-29T12:00:00Z"),
            ],
            "metric": ["day_ahead_price"] * 2,
            "value": [40.0, 50.0],
            "resolution": ["PT60M"] * 2,
        }
    )
    result = build_market_features(rows, market_config())
    assert result.loc[1, "price_lag_24h"] == 40.0
    assert result.loc[0, "hour_of_day"] == 13
    assert result.loc[1, "hour_of_day"] == 14


def test_backward_alignment_never_uses_future_and_stops_at_tolerance() -> None:
    battery = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:05Z", "2026-01-01T00:10Z", "2026-01-01T00:21Z"]
            ),
            "asset_id": ["BESS-001"] * 3,
            "some_battery_feature": [1.0, 2.0, 3.0],
        }
    )
    market = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:10Z"]),
            "price_eur_per_mwh": [1.0, 2.0],
        }
    )
    result = align_market_to_battery(battery, market, market_config())
    assert result.loc[0, "price_eur_per_mwh"] == 1.0
    assert result.loc[0, "market_data_age_minutes"] == 5.0
    assert result.loc[1, "price_eur_per_mwh"] == 2.0
    assert pd.isna(result.loc[2, "price_eur_per_mwh"])
    assert not result.loc[2, "market_data_available"]
    assert str(result["timestamp_utc"].dt.tz) == "UTC"


def test_alignment_expires_at_source_interval_boundary() -> None:
    battery = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:14:59Z", "2026-01-01T00:15:00Z"]),
            "asset_id": ["BESS-001"] * 2,
        }
    )
    market = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00Z"]),
            "price_eur_per_mwh": [50.0],
            "market_source_resolution_minutes": [15.0],
        }
    )
    config = market_config(market_alignment_tolerance="60min")
    result = align_market_to_battery(battery, market, config)
    assert result.loc[0, "price_eur_per_mwh"] == 50.0
    assert result.loc[0, "market_data_available"]
    assert pd.isna(result.loc[1, "price_eur_per_mwh"])
    assert not result.loc[1, "market_data_available"]


def test_market_generation_is_deterministic_and_provenance_is_derived() -> None:
    first = build_market_features(price_observations(5), market_config())
    second = build_market_features(price_observations(5), market_config())
    pd.testing.assert_frame_equal(first, second)
    assert set(first["data_provenance"]) == {"DERIVED"}
    assert first.attrs["column_provenance"]["price_eur_per_mwh"] == "REAL"
