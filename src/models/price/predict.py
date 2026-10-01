"""Contract-checked timestamp-aligned price inference."""

import numpy as np
import pandas as pd

from models.artifact import ModelArtifact
from models.price.config import PriceModelConfig
from models.price.dataset import build_price_features, validate_price_predictors


def predict_prices(
    artifact: ModelArtifact,
    observations: pd.DataFrame,
    target_timestamps: pd.Series | pd.DatetimeIndex,
    forecast_origin_utc: pd.Timestamp,
    *,
    config: PriceModelConfig | None = None,
    future_forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = config or PriceModelConfig()
    features = build_price_features(
        observations,
        target_timestamps,
        forecast_origin_utc=forecast_origin_utc,
        config=cfg,
        future_forecasts=future_forecasts,
    )
    validate_price_predictors(list(artifact.feature_names))
    missing = sorted(set(artifact.feature_names) - set(features.columns))
    if missing:
        raise ValueError(f"Required price features are missing: {missing}")
    X = features[list(artifact.feature_names)]
    if np.isinf(X.select_dtypes(include=[np.number]).to_numpy(dtype=float)).any():
        raise ValueError("Unsupported non-finite price input")
    required_lags = [name for name in artifact.feature_names if name.startswith("price_lag_")]
    if X[required_lags].isna().any().any():
        raise ValueError("Required exact timestamp lag is unavailable")
    prediction = np.asarray(artifact.estimator.predict(X), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Price model produced non-finite output")
    origin = pd.Timestamp(forecast_origin_utc).tz_convert("UTC")
    result = pd.DataFrame(
        {
            "timestamp_utc": features["timestamp_utc"],
            "market_region": cfg.market_region,
            "forecast_origin_utc": origin,
            "target_timestamp_utc": features["timestamp_utc"],
            "predicted_price_eur_per_mwh": prediction,
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "feature_set_version": artifact.metadata["feature_set_version"],
            "data_provenance": "MODEL_PREDICTION",
        }
    )
    result["horizon_minutes"] = (
        (result["target_timestamp_utc"] - origin).dt.total_seconds() / 60
    ).astype(int)
    return result
