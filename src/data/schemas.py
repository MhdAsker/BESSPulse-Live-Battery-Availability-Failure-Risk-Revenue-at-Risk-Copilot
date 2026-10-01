"""Typed contracts for normalized ENTSO-E data and ingestion summaries."""

import math
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Dataset(StrEnum):
    DAY_AHEAD_PRICES = "day_ahead_prices"
    ACTUAL_LOAD = "actual_load"
    LOAD_FORECAST = "load_forecast"
    ACTUAL_GENERATION = "actual_generation"
    GENERATION_FORECAST = "generation_forecast"


class MarketMetric(StrEnum):
    DAY_AHEAD_PRICE = "day_ahead_price"
    ACTUAL_LOAD = "actual_load"
    LOAD_FORECAST = "load_forecast"
    WIND_ONSHORE_GENERATION = "wind_onshore_generation"
    WIND_OFFSHORE_GENERATION = "wind_offshore_generation"
    SOLAR_GENERATION = "solar_generation"
    WIND_ONSHORE_GENERATION_FORECAST = "wind_onshore_generation_forecast"
    WIND_OFFSHORE_GENERATION_FORECAST = "wind_offshore_generation_forecast"
    SOLAR_GENERATION_FORECAST = "solar_generation_forecast"
    WIND_GENERATION = "wind_generation"
    WIND_GENERATION_FORECAST = "wind_generation_forecast"


class SeriesType(StrEnum):
    DAY_AHEAD = "DAY_AHEAD"
    ACTUAL = "ACTUAL"
    FORECAST = "FORECAST"
    DERIVED = "DERIVED"


class MarketObservation(BaseModel):
    """One normalized long-form observation with source lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    market_region: str = Field(min_length=1)
    metric: MarketMetric
    value: float
    unit: str = Field(min_length=1)
    series_type: SeriesType
    source: str = "ENTSO-E Transparency Platform"
    data_provenance: str = "REAL"
    retrieved_at_utc: datetime
    raw_content_hash: str = Field(min_length=64, max_length=64)
    document_id: str | None = None
    timeseries_mrid: str = ""
    business_type: str | None = None
    process_type: str | None = None
    psr_type: str | None = None
    resolution: str

    @field_validator("timestamp_utc", "retrieved_at_utc")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset != timedelta(0):
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @field_validator("data_provenance")
    @classmethod
    def known_provenance(cls, value: str) -> str:
        if value not in {"REAL", "DERIVED"}:
            raise ValueError("market provenance must be REAL or DERIVED")
        return value


class RawArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    xml_path: Path
    metadata_path: Path
    content_hash: str
    created: bool


class FetchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: Dataset
    observations: tuple[MarketObservation, ...]
    raw_artifact: RawArtifact


class ContinuityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_interval: timedelta
    expected_observation_count: int
    actual_observation_count: int
    missing_timestamps: tuple[datetime, ...]
    duplicate_timestamps: tuple[datetime, ...]


class DatasetFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: Dataset
    error_type: str
    message: str


class IngestionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_start: datetime
    requested_end: datetime
    market_region: str
    datasets_requested: tuple[Dataset, ...]
    datasets_succeeded: tuple[Dataset, ...]
    datasets_unavailable: tuple[Dataset, ...]
    datasets_failed: tuple[DatasetFailure, ...]
    observations_written: int
    raw_files_written: int
    validation_warnings: tuple[str, ...]
