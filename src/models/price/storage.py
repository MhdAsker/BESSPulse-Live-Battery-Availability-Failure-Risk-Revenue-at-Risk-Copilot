"""Idempotent persistence for point and quantile price predictions."""

from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import PricePredictionModel


def store_price_predictions(
    session: Session, predictions: pd.DataFrame, *, quantile: float | None = None
) -> int:
    required = {
        "forecast_origin_utc",
        "target_timestamp_utc",
        "market_region",
        "model_name",
        "model_version",
        "predicted_price_eur_per_mwh",
        "horizon_minutes",
        "feature_set_version",
        "data_provenance",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing price prediction fields: {sorted(missing)}")
    quantile_key = -1.0 if quantile is None else quantile
    written = 0
    for raw in predictions.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        origin = pd.Timestamp(row["forecast_origin_utc"]).to_pydatetime()
        target = pd.Timestamp(row["target_timestamp_utc"]).to_pydatetime()
        identity = select(PricePredictionModel).where(
            PricePredictionModel.forecast_origin_utc == origin,
            PricePredictionModel.target_timestamp_utc == target,
            PricePredictionModel.market_region == row["market_region"],
            PricePredictionModel.model_version == row["model_version"],
            PricePredictionModel.quantile == quantile_key,
        )
        existing = session.scalar(identity)
        values = {
            "model_name": str(row["model_name"]),
            "predicted_price_eur_per_mwh": float(row["predicted_price_eur_per_mwh"]),
            "horizon_minutes": int(row["horizon_minutes"]),
            "feature_set_version": str(row["feature_set_version"]),
            "data_provenance": str(row["data_provenance"]),
        }
        if existing is not None:
            changed = any(getattr(existing, key) != value for key, value in values.items())
            if not changed:
                continue
            for key, value in values.items():
                setattr(existing, key, value)
            existing.created_at = datetime.now(UTC)
        else:
            session.add(
                PricePredictionModel(
                    forecast_origin_utc=origin,
                    target_timestamp_utc=target,
                    market_region=str(row["market_region"]),
                    model_version=str(row["model_version"]),
                    quantile=quantile_key,
                    created_at=datetime.now(UTC),
                    **values,
                )
            )
        session.flush()
        written += 1
    session.commit()
    return written
