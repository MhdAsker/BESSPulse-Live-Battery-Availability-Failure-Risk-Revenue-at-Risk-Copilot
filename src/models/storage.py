"""Persistence helper for expected-behavior predictions and residuals."""

from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import ModelPredictionModel
from features.registry import FEATURE_SET_VERSION


def persist_predictions(
    session: Session,
    frame: pd.DataFrame,
    *,
    entity_column: str,
    prediction_column: str,
    model_name: str,
    model_version: str,
    actual_column: str | None = None,
    residual_column: str | None = None,
) -> int:
    written = 0
    for row in frame.to_dict(orient="records"):
        identity = select(ModelPredictionModel.id).where(
            ModelPredictionModel.timestamp_utc == row["timestamp_utc"],
            ModelPredictionModel.entity_id == str(row[entity_column]),
            ModelPredictionModel.model_version == model_version,
            ModelPredictionModel.prediction_name == prediction_column,
        )
        if session.scalar(identity) is not None:
            continue
        actual = cast(Any, row.get(actual_column)) if actual_column else None
        residual = cast(Any, row.get(residual_column)) if residual_column else None
        session.add(
            ModelPredictionModel(
                timestamp_utc=row["timestamp_utc"],
                entity_id=str(row[entity_column]),
                model_name=model_name,
                model_version=model_version,
                prediction_name=prediction_column,
                prediction_value=float(row[prediction_column]),
                actual_value=float(actual) if pd.notna(actual) else None,
                residual_value=float(residual) if pd.notna(residual) else None,
                prediction_provenance="MODEL_PREDICTION",
                residual_provenance="DERIVED" if pd.notna(residual) else None,
                feature_set_version=FEATURE_SET_VERSION,
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        written += 1
    session.commit()
    return written


def persist_delivery_risk_predictions(
    session: Session,
    frame: pd.DataFrame,
    *,
    horizon_hours: int,
    model_name: str,
    model_version: str,
    decision_threshold: float = 0.5,
) -> int:
    """Persist probabilities without pretending future outcomes are known at inference."""

    probability_column = f"failure_probability_{horizon_hours}h"
    required = ["timestamp_utc", "asset_id", probability_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing delivery-risk prediction columns: {missing}")
    written = 0
    for row in frame.to_dict(orient="records"):
        identity = select(ModelPredictionModel.id).where(
            ModelPredictionModel.timestamp_utc == row["timestamp_utc"],
            ModelPredictionModel.entity_id == str(row["asset_id"]),
            ModelPredictionModel.model_version == model_version,
            ModelPredictionModel.prediction_name == probability_column,
        )
        if session.scalar(identity) is not None:
            continue
        probability = float(row[probability_column])
        session.add(
            ModelPredictionModel(
                timestamp_utc=row["timestamp_utc"],
                entity_id=str(row["asset_id"]),
                model_name=model_name,
                model_version=model_version,
                prediction_name=probability_column,
                prediction_value=probability,
                actual_value=None,
                residual_value=None,
                prediction_horizon_hours=horizon_hours,
                decision_threshold=decision_threshold,
                predicted_class=probability >= decision_threshold,
                prediction_provenance="MODEL_PREDICTION",
                residual_provenance=None,
                feature_set_version=FEATURE_SET_VERSION,
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        written += 1
    session.commit()
    return written
