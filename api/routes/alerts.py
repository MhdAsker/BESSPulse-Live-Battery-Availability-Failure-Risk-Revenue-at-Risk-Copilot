"""Filtered, bounded alert history endpoint."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from alerts.schemas import AlertStatus, AlertType
from api.config import APISettings
from api.dependencies import get_alert_service, get_api_settings
from api.errors import APIError
from api.routes.racks import _validate_range
from api.schemas.alerts import AlertPage
from api.services import AlertQueryService

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=AlertPage, summary="Filter and paginate decision-support alerts")
def alerts(
    service: Annotated[AlertQueryService, Depends(get_alert_service)],
    settings: Annotated[APISettings, Depends(get_api_settings)],
    asset_id: str | None = None,
    component_id: str | None = None,
    status: AlertStatus | None = None,
    priority_level: str | None = None,
    alert_type: AlertType | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=50, ge=1),
    offset: int = Query(default=0, ge=0),
) -> AlertPage:
    _validate_range(start, end, settings)
    if limit > settings.api_max_page_size:
        raise APIError(422, "page_size_exceeded", f"limit must be <= {settings.api_max_page_size}")
    return service.list(
        asset_id=asset_id,
        component_id=component_id,
        status=status.value if status else None,
        priority_level=priority_level,
        alert_type=alert_type.value if alert_type else None,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )
