"""Cheap liveness and dependency readiness endpoints."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from api import __version__
from api.dependencies import get_health_service
from api.schemas.health import HealthResponse
from api.services import HealthService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Combined health status")
def health(service: Annotated[HealthService, Depends(get_health_service)]) -> HealthResponse:
    components = service.components()
    status = (
        "healthy"
        if all(value == "available" for key, value in components.items() if key != "copilot")
        else "degraded"
    )
    readiness = "ready" if components["database"] == "available" else "not_ready"
    return HealthResponse(
        status=status,
        version=__version__,
        liveness="alive",
        readiness=readiness,
        database_status=components["database"],
        model_status=components["delivery_risk"],
        market_data_status=components["market_data"],
        components=components,
        timestamp_utc=datetime.now(UTC),
    )


@router.get("/live", response_model=HealthResponse, summary="Process liveness")
def live() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=__version__,
        liveness="alive",
        readiness="unknown",
        database_status="not_checked",
        model_status="not_checked",
        market_data_status="not_checked",
        components={"process": "available"},
        timestamp_utc=datetime.now(UTC),
    )


@router.get("/ready", response_model=HealthResponse, summary="Core dependency readiness")
def ready(service: Annotated[HealthService, Depends(get_health_service)]) -> HealthResponse:
    return health(service)
