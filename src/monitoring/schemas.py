"""Typed monitoring snapshots."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthStatus(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class MonitoringSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    observed_at_utc: datetime
    metric_scope: str
    subsystem: str = "general"
    metric_name: str
    metric_value: float | None = None
    threshold: float | None = None
    status: HealthStatus
    model_name: str | None = None
    model_version: str | None = None
    asset_id: str | None = None
    reference_start: datetime | None = None
    reference_end: datetime | None = None
    current_start: datetime | None = None
    current_end: datetime | None = None
    sample_count: int = 0
    details: dict[str, Any] = {}
    data_provenance: str = "DERIVED"
