"""Explicitly development/demo-only simulation controls."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.config import APISettings
from api.dependencies import get_api_settings, get_simulation_service
from api.errors import SimulationDisabledError
from api.schemas.simulation import (
    FaultInjectionRequest,
    FaultInjectionResponse,
    SimulationResetRequest,
    SimulationResetResponse,
)
from api.services import SimulationService

router = APIRouter(prefix="/simulation", tags=["simulation controls (development/demo)"])


def _require_enabled(settings: APISettings) -> None:
    if not settings.enable_simulation_control_api:
        raise SimulationDisabledError


@router.post(
    "/fault",
    response_model=FaultInjectionResponse,
    status_code=201,
    summary="Schedule a validated simulated fault",
)
def inject_fault(
    request: FaultInjectionRequest,
    service: Annotated[SimulationService, Depends(get_simulation_service)],
    settings: Annotated[APISettings, Depends(get_api_settings)],
) -> FaultInjectionResponse:
    _require_enabled(settings)
    return service.inject(request)


@router.post(
    "/reset",
    response_model=SimulationResetResponse,
    summary="Reset in-memory simulation control state",
)
def reset_simulation(
    request: SimulationResetRequest,
    service: Annotated[SimulationService, Depends(get_simulation_service)],
    settings: Annotated[APISettings, Depends(get_api_settings)],
) -> SimulationResetResponse:
    _require_enabled(settings)
    return service.reset(request)
