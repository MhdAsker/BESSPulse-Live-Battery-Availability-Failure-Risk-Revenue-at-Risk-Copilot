from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from models.price.dataset import (
    TARGET_COLUMN,
    build_price_dataset,
    build_price_features,
    chronological_price_split,
    validate_price_predictors,
)


def prices(periods: int = 1000, start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    values = 50 + 20 * np.sin(np.arange(periods) * 2 * np.pi / 24)
    if periods > 300:
        values[300] = -25
    if periods > 500:
        values[500] = 1000
    return pd.DataFrame({"timestamp_utc": timestamps, TARGET_COLUMN: values})


def test_timestamp_lags_are_exact_and_missing_is_nan() -> None:
    source = prices(200).drop(index=[8, 175]).reset_index(drop=True)
    features = build_price_features(source)
    row = features.loc[features["timestamp_utc"] == pd.Timestamp("2024-01-08 08:00Z")].iloc[0]
    expected = source.set_index("timestamp_utc").loc[
        pd.Timestamp("2024-01-07 08:00Z"), TARGET_COLUMN
    ]
    assert row["price_lag_24h"] == pytest.approx(expected)
    assert np.isnan(row["price_lag_168h"])
    missing = features.loc[features["timestamp_utc"] == pd.Timestamp("2024-01-09 07:00Z")].iloc[0]
    assert np.isnan(missing["price_lag_24h"])


def test_lag_is_not_row_count_and_never_uses_future() -> None:
    source = prices(220).drop(index=[100, 101]).reset_index(drop=True)
    features = build_price_features(source)
    target = source.loc[150, "timestamp_utc"]
    expected_time = target - pd.Timedelta("24h")
    expected = source.set_index("timestamp_utc").loc[expected_time, TARGET_COLUMN]
    actual = features.set_index("timestamp_utc").loc[target, "price_lag_24h"]
    assert actual == expected
    changed = source.copy()
    changed.loc[changed["timestamp_utc"] > target, TARGET_COLUMN] = 99999
    recomputed = build_price_features(changed).set_index("timestamp_utc").loc[target]
    original = features.set_index("timestamp_utc").loc[target]
    pd.testing.assert_series_equal(original, recomputed)


def test_24h_horizon_rolling_features_ignore_post_origin_actuals() -> None:
    source = prices(400)
    origin = source.loc[300, "timestamp_utc"]
    target = pd.DatetimeIndex([origin + pd.Timedelta("12h")])
    baseline = build_price_features(source, target).iloc[0]
    changed = source.copy()
    changed.loc[changed["timestamp_utc"] > origin, TARGET_COLUMN] = 99999
    recomputed = build_price_features(changed, target).iloc[0]
    columns = [
        column
        for column in baseline.index
        if column.startswith("rolling_price_") or column.startswith("price_volatility_")
    ]
    pd.testing.assert_series_equal(baseline[columns], recomputed[columns])


@pytest.mark.parametrize("date", ["2024-03-31", "2024-10-27"])
def test_dst_calendar_and_lags_use_utc_alignment(date: str) -> None:
    timestamps = pd.date_range(f"{date} 00:00", periods=200, freq="h", tz="UTC")
    source = pd.DataFrame({"timestamp_utc": timestamps, TARGET_COLUMN: np.arange(200.0)})
    features = build_price_features(source)
    local = features["timestamp_utc"].dt.tz_convert(ZoneInfo("Europe/Berlin"))
    assert features["hour_of_day"].tolist() == local.dt.hour.tolist()
    assert features.loc[24, "price_lag_24h"] == 0


def test_dst_missing_and_repeated_local_hours_are_explicit() -> None:
    spring = pd.Series(pd.date_range("2024-03-30 22:00", periods=8, freq="h", tz="UTC"))
    spring_hours = spring.dt.tz_convert("Europe/Berlin").dt.hour.tolist()
    assert 2 not in spring_hours
    autumn = pd.Series(pd.date_range("2024-10-26 22:00", periods=8, freq="h", tz="UTC"))
    local = autumn.dt.tz_convert("Europe/Berlin")
    repeated = local[local.dt.hour == 2]
    assert len(repeated) == 2
    assert {value.utcoffset().total_seconds() / 3600 for value in repeated} == {1.0, 2.0}


def test_chronological_split_and_minimum_history() -> None:
    dataset = build_price_dataset(prices())
    split = chronological_price_split(dataset)
    assert split.train.timestamps.max() < split.validation.timestamps.min()
    assert split.validation.timestamps.max() < split.test.timestamps.min()
    with pytest.raises(ValueError, match="Insufficient price history"):
        build_price_dataset(prices(300))


@pytest.mark.parametrize(
    "column",
    [
        "target_price",
        "future_price_1h",
        "future_actual_load",
        "site_soc",
        "battery_power_mw",
        "technical_availability",
        "fault_id",
        "anomaly_score",
        "delivery_risk_probability",
    ],
)
def test_leakage_and_battery_fields_are_rejected(column: str) -> None:
    with pytest.raises(ValueError, match="leakage-prohibited"):
        validate_price_predictors(["price_lag_24h", column])


def test_negative_and_extreme_prices_are_preserved() -> None:
    dataset = build_price_dataset(prices())
    assert -25 in dataset.target.to_numpy()
    assert 1000 in dataset.target.to_numpy()
