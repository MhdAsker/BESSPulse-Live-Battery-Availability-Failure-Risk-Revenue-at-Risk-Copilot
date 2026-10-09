"""Monitoring API contracts."""

from datetime import datetime
from typing import Any

from api.schemas.common import APIModel
from monitoring.schemas import HealthStatus


class MonitoringMetricResponse(APIModel):
    observed_at_utc: datetime
    metric_scope: str
    subsystem: str
    metric_name: str
    metric_value: float | None
    threshold: float | None
    status: HealthStatus
    model_name: str | None
    model_version: str | None
    asset_id: str | None
    reference_start: datetime | None
    reference_end: datetime | None
    current_start: datetime | None
    current_end: datetime | None
    sample_count: int
    details: dict[str, Any]
    data_provenance: str


class MonitoringResponse(APIModel):
    items: tuple[MonitoringMetricResponse, ...]
