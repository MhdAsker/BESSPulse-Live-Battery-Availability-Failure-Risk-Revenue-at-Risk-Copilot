"""Delivery-risk inference response contract."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, Freshness, Provenance


class CalibrationMetadata(APIModel):
    horizon_hours: int
    status: str
    limitation: str | None = None


class DeliveryRiskResponse(APIModel):
    asset_id: str
    timestamp_utc: datetime
    failure_probability_6h: float = Field(ge=0, le=1)
    failure_probability_12h: float = Field(ge=0, le=1)
    failure_probability_24h: float = Field(ge=0, le=1)
    model_version_6h: str
    model_version_12h: str
    model_version_24h: str
    feature_set_version: str
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    calibration: tuple[CalibrationMetadata, ...]
    freshness: Freshness
    data_provenance: Provenance
