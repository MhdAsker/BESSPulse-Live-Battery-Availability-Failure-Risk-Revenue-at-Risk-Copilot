"""Causal price feature construction, coverage reporting, and temporal splitting."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from models.price.config import PriceModelConfig

TARGET_COLUMN = "day_ahead_price_eur_per_mwh"
PRICE_FEATURES = (
    "price_lag_24h",
    "price_lag_168h",
    "rolling_price_mean_24h",
    "rolling_price_mean_7d",
    "price_volatility_24h",
    "price_volatility_7d",
    "hour_of_day",
    "day_of_week",
    "hour_of_week",
    "is_weekend",
    "utc_offset_hours",
)
OPTIONAL_FORECAST_FEATURES = (
    "load_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "renewable_forecast_mw",
)

PROHIBITED_EXACT = frozenset(
    {
        TARGET_COLUMN,
        "target",
        "target_price",
        "future_price",
        "future_actual_load",
        "future_actual_wind",
        "future_actual_solar",
        "soc",
        "site_soc",
        "battery_power",
        "availability",
        "faults",
        "anomalies",
        "delivery_risk_probability",
    }
)


def validate_price_predictors(columns: list[str] | tuple[str, ...]) -> None:
    """Reject target, future actual, battery, reliability, and outcome-derived fields."""

    violations: list[str] = []
    forbidden_fragments = (
        "battery",
        "soc",
        "availability",
        "fault",
        "anomaly",
        "delivery_risk",
        "future_residual",
        "post_delivery",
        "price_t_plus_",
    )
    for column in columns:
        normalized = column.lower()
        if (
            normalized in PROHIBITED_EXACT
            or normalized.startswith("future_actual_")
            or normalized.startswith("future_price")
            or normalized.startswith("target_price")
            or any(fragment in normalized for fragment in forbidden_fragments)
        ):
            violations.append(column)
    if violations:
        raise ValueError(f"Price-model leakage-prohibited features: {sorted(set(violations))}")


def _utc_timestamps(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("Price timestamps cannot be missing")
    for value in values:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Price timestamps must be timezone-aware")
    return pd.to_datetime(values, utc=True)


def normalize_price_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Normalize long-form ENTSO-E rows or an explicit timestamp/price frame."""

    if "metric" in observations:
        selected = observations.loc[
            observations["metric"].map(lambda value: getattr(value, "value", value))
            == "day_ahead_price"
        ].copy()
        result = selected.rename(columns={"value": TARGET_COLUMN})
    else:
        result = observations.copy()
        if TARGET_COLUMN not in result and "price_eur_per_mwh" in result:
            result = result.rename(columns={"price_eur_per_mwh": TARGET_COLUMN})
    required = {"timestamp_utc", TARGET_COLUMN}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Missing price observation fields: {sorted(missing)}")
    result["timestamp_utc"] = _utc_timestamps(result["timestamp_utc"])
    if result.duplicated("timestamp_utc").any():
        raise ValueError("Duplicate price timestamps require an explicit revision-selection policy")
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="raise")
    if not np.isfinite(result[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Prices must be finite")
    keep = ["timestamp_utc", TARGET_COLUMN]
    for column in ["resolution", *OPTIONAL_FORECAST_FEATURES]:
        if column in result:
            keep.append(column)
    return result[keep].sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)


def infer_resolution(prices: pd.DataFrame) -> pd.Timedelta:
    timestamps = pd.DatetimeIndex(prices["timestamp_utc"].sort_values().unique())
    if len(timestamps) < 2:
        raise ValueError("At least two observations are required to infer resolution")
    differences = pd.Series(timestamps[1:] - timestamps[:-1])
    positive = differences[differences > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("Could not infer a positive market resolution")
    return pd.Timedelta(positive.mode().iloc[0])


def _exact_lookup(series: pd.Series, targets: pd.DatetimeIndex) -> NDArray[np.float64]:
    return series.reindex(targets).to_numpy(dtype=float)


def build_price_features(
    observations: pd.DataFrame,
    target_timestamps: pd.Series | pd.DatetimeIndex | None = None,
    *,
    forecast_origin_utc: pd.Timestamp | None = None,
    config: PriceModelConfig | None = None,
    future_forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build timestamp-addressed predictors using only observations available by origin."""

    cfg = config or PriceModelConfig()
    prices = normalize_price_observations(observations)
    if forecast_origin_utc is not None:
        origin = pd.Timestamp(forecast_origin_utc)
        if origin.tzinfo is None:
            raise ValueError("forecast_origin_utc must be timezone-aware")
        origin = origin.tz_convert("UTC")
        prices = prices.loc[prices["timestamp_utc"] <= origin].copy()
    if target_timestamps is None:
        targets = pd.DatetimeIndex(normalize_price_observations(observations)["timestamp_utc"])
    else:
        target_series = pd.Series(target_timestamps)
        targets = pd.DatetimeIndex(_utc_timestamps(target_series))
    if targets.has_duplicates:
        raise ValueError("Target timestamps must be unique")
    if forecast_origin_utc is not None and (targets <= origin).any():
        raise ValueError("Every forecast target must be strictly after forecast origin")
    history = pd.Series(
        prices[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(prices["timestamp_utc"]),
    )
    output = pd.DataFrame({"timestamp_utc": targets})
    output["price_lag_24h"] = _exact_lookup(history, targets - pd.Timedelta("24h"))
    output["price_lag_168h"] = _exact_lookup(history, targets - pd.Timedelta("168h"))
    # Simulate a 24-hour-ahead feature availability boundary for every supervised
    # row. A conventional window ending at T would contain prices learned after a
    # rolling origin O for targets in (O, O+24h], so all rolling summaries end at
    # T-24h instead. This is conservative for shorter horizons and causal for all.
    cutoffs = targets - pd.Timedelta(hours=cfg.forecast_horizon_hours)
    for label, window in (("24h", pd.Timedelta("24h")), ("7d", pd.Timedelta("168h"))):
        rolling = history.rolling(window=window, closed="right")
        output[f"rolling_price_mean_{label}"] = rolling.mean().reindex(cutoffs).to_numpy()
        output[f"price_volatility_{label}"] = rolling.std().reindex(cutoffs).to_numpy()
    local = output["timestamp_utc"].dt.tz_convert(cfg.market_timezone)
    output["hour_of_day"] = local.dt.hour.astype(int)
    output["day_of_week"] = local.dt.dayofweek.astype(int)
    output["hour_of_week"] = output["day_of_week"] * 24 + output["hour_of_day"]
    output["is_weekend"] = (output["day_of_week"] >= 5).astype(int)
    output["utc_offset_hours"] = local.map(lambda value: value.utcoffset().total_seconds() / 3600)
    if future_forecasts is not None:
        known = future_forecasts.copy()
        known["timestamp_utc"] = _utc_timestamps(known["timestamp_utc"])
        allowed = [column for column in OPTIONAL_FORECAST_FEATURES if column in known]
        output = output.merge(known[["timestamp_utc", *allowed]], on="timestamp_utc", how="left")
    validate_price_predictors(
        [str(column) for column in output.columns if column != "timestamp_utc"]
    )
    numeric = output.drop(columns="timestamp_utc").select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Price predictors contain infinity")
    return output


@dataclass(frozen=True)
class PriceDataset:
    timestamps: pd.Series
    predictors: pd.DataFrame
    target: pd.Series
    dataset_hash: str

    def subset(self, positions: NDArray[np.intp]) -> "PriceDataset":
        return PriceDataset(
            self.timestamps.iloc[positions].reset_index(drop=True),
            self.predictors.iloc[positions].reset_index(drop=True),
            self.target.iloc[positions].reset_index(drop=True),
            self.dataset_hash,
        )


@dataclass(frozen=True)
class PriceSplit:
    train: PriceDataset
    validation: PriceDataset
    test: PriceDataset


def _hash_dataset(frame: pd.DataFrame) -> str:
    schema = "|".join(f"{name}:{frame[name].dtype}" for name in frame)
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(schema.encode() + values).hexdigest()


def build_price_dataset(
    observations: pd.DataFrame,
    config: PriceModelConfig | None = None,
    *,
    include_system_forecasts: bool = False,
) -> PriceDataset:
    cfg = config or PriceModelConfig()
    prices = normalize_price_observations(observations)
    feature_source = (
        prices
        if include_system_forecasts
        else prices.drop(
            columns=[column for column in OPTIONAL_FORECAST_FEATURES if column in prices]
        )
    )
    future = feature_source[
        ["timestamp_utc", *[c for c in OPTIONAL_FORECAST_FEATURES if c in feature_source]]
    ]
    features = build_price_features(
        feature_source,
        config=cfg,
        future_forecasts=future if include_system_forecasts else None,
    )
    merged = prices[["timestamp_utc", TARGET_COLUMN]].merge(
        features, on="timestamp_utc", how="inner", validate="one_to_one"
    )
    predictors = [*PRICE_FEATURES]
    if include_system_forecasts:
        predictors.extend(column for column in OPTIONAL_FORECAST_FEATURES if column in merged)
    required = ["price_lag_24h", "price_lag_168h"]
    usable = merged.dropna(subset=[TARGET_COLUMN, *required]).reset_index(drop=True)
    if usable.empty:
        raise ValueError("No rows remain after required timestamp-lag history")
    duration = usable["timestamp_utc"].max() - prices["timestamp_utc"].min()
    if duration < pd.Timedelta(hours=cfg.minimum_training_history_hours):
        raise ValueError(
            "Insufficient price history: "
            f"{duration.total_seconds() / 3600:.1f}h < {cfg.minimum_training_history_hours}h"
        )
    X = usable[predictors].copy()
    validate_price_predictors(list(X.columns))
    hashed = usable[["timestamp_utc", *predictors, TARGET_COLUMN]]
    return PriceDataset(usable["timestamp_utc"], X, usable[TARGET_COLUMN], _hash_dataset(hashed))


def chronological_price_split(
    dataset: PriceDataset, config: PriceModelConfig | None = None
) -> PriceSplit:
    cfg = config or PriceModelConfig()
    count = len(dataset.timestamps)
    if count < 10:
        raise ValueError("Too few price rows for chronological split")
    train_end = max(1, int(count * cfg.train_fraction))
    validation_end = max(train_end + 1, int(count * (cfg.train_fraction + cfg.validation_fraction)))
    validation_end = min(validation_end, count - 1)
    split = PriceSplit(
        dataset.subset(np.arange(0, train_end)),
        dataset.subset(np.arange(train_end, validation_end)),
        dataset.subset(np.arange(validation_end, count)),
    )
    if not (
        split.train.timestamps.max()
        < split.validation.timestamps.min()
        <= split.validation.timestamps.max()
        < split.test.timestamps.min()
    ):
        raise AssertionError("Chronological price split ordering failed")
    return split


def coverage_report(
    observations: pd.DataFrame, config: PriceModelConfig | None = None
) -> dict[str, Any]:
    prices = normalize_price_observations(observations)
    resolution = infer_resolution(prices)
    expected = pd.date_range(
        prices["timestamp_utc"].min(), prices["timestamp_utc"].max(), freq=resolution
    )
    actual = pd.DatetimeIndex(prices["timestamp_utc"])
    features = build_price_features(prices, config=config)
    return {
        "start_utc": prices["timestamp_utc"].min().isoformat(),
        "end_utc": prices["timestamp_utc"].max().isoformat(),
        "observations": len(prices),
        "resolution_minutes": int(resolution.total_seconds() / 60),
        "missing_intervals": len(expected.difference(actual)),
        "duplicate_intervals": int(prices.duplicated("timestamp_utc").sum()),
        "negative_price_observations": int((prices[TARGET_COLUMN] < 0).sum()),
        "feature_missingness": {
            column: int(features[column].isna().sum())
            for column in features
            if column != "timestamp_utc"
        },
        "rows_after_lags": int(
            features[["price_lag_24h", "price_lag_168h"]].notna().all(axis=1).sum()
        ),
    }
