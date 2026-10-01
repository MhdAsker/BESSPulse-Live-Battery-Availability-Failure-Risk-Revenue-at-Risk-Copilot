"""Reusable API response contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ProvenanceType(StrEnum):
    REAL = "REAL"
    SIMULATED = "SIMULATED"
    DERIVED = "DERIVED"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    DECISION_SUPPORT = "DECISION_SUPPORT"


class Provenance(APIModel):
    provenance_type: ProvenanceType
    source: str
    source_version: str
    generated_at_utc: datetime
    model_version: str | None = None
    feature_set_version: str | None = None

    @field_validator("generated_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return value


class Freshness(APIModel):
    as_of_utc: datetime
    age_seconds: float = Field(ge=0)
    is_stale: bool


class ErrorResponse(APIModel):
    error_code: str
    message: str
    details: Any | None = None
    request_id: str | None = None
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PageMetadata(APIModel):
    total: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)
