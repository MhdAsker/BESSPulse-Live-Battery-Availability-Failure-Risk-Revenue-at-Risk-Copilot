"""Filtered model and data-health observations."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_monitoring_service
from api.schemas.monitoring import MonitoringMetricResponse, MonitoringResponse
from monitoring.schemas import HealthStatus
from monitoring.service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("", response_model=MonitoringResponse, summary="Query model and data health")
def monitoring(
    service: Annotated[MonitoringService, Depends(get_monitoring_service)],
    scope: str | None = None,
    model_name: str | None = None,
    asset_id: str | None = None,
    status: HealthStatus | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> MonitoringResponse:
    snapshots = service.list(
        scope=scope, model_name=model_name, asset_id=asset_id, status=status, limit=limit
    )
    return MonitoringResponse(
        items=tuple(MonitoringMetricResponse.model_validate(item) for item in snapshots)
    )
