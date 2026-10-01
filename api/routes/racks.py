"""Rack state and anomaly history endpoints."""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.config import APISettings
from api.dependencies import get_api_settings, get_rack_service
from api.errors import APIError
from api.schemas.anomalies import AnomalyPage
from api.schemas.racks import RackResponse
from api.services import RackService

router = APIRouter(prefix="/racks", tags=["racks"])


@router.get("/{rack_id}", response_model=RackResponse, summary="Latest observable rack state")
def rack(rack_id: str, service: Annotated[RackService, Depends(get_rack_service)]) -> RackResponse:
    return service.latest(rack_id)


@router.get(
    "/{rack_id}/anomalies", response_model=AnomalyPage, summary="Paginated rack anomaly events"
)
def rack_anomalies(
    rack_id: str,
    service: Annotated[RackService, Depends(get_rack_service)],
    settings: Annotated[APISettings, Depends(get_api_settings)],
    start: datetime | None = None,
    end: datetime | None = None,
    detector: str | None = None,
    active_only: bool = False,
    limit: int = Query(default=50, ge=1),
    offset: int = Query(default=0, ge=0),
) -> AnomalyPage:
    _validate_range(start, end, settings)
    if limit > settings.api_max_page_size:
        raise APIError(422, "page_size_exceeded", f"limit must be <= {settings.api_max_page_size}")
    return service.anomalies(
        rack_id,
        start=start,
        end=end,
        detector=detector,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )


def _validate_range(start: datetime | None, end: datetime | None, settings: APISettings) -> None:
    for name, value in (("start", start), ("end", end)):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise APIError(422, "invalid_datetime", f"{name} must be timezone-aware")
    if start is not None and end is not None:
        if start >= end:
            raise APIError(422, "invalid_date_range", "start must be before end")
        if end - start > timedelta(days=settings.api_max_query_days):
            raise APIError(
                422,
                "date_range_too_large",
                f"date range must not exceed {settings.api_max_query_days} days",
            )
