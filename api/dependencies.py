"""Request-scoped database and service dependency wiring."""

from collections.abc import Generator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from api.config import APISettings
from api.services import (
    AlertQueryService,
    AssetService,
    AvailabilityService,
    CommercialService,
    CopilotService,
    HealthService,
    MarketService,
    RackService,
    RiskService,
    SimulationService,
)
from monitoring.service import MonitoringService


def get_api_settings(request: Request) -> APISettings:
    return cast(APISettings, request.app.state.settings)


def get_db_session(request: Request) -> Generator[Session, None, None]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DBSession = Annotated[Session, Depends(get_db_session)]
Settings = Annotated[APISettings, Depends(get_api_settings)]


def get_asset_service(session: DBSession, settings: Settings) -> AssetService:
    return AssetService(session, settings)


def get_availability_service(session: DBSession, settings: Settings) -> AvailabilityService:
    return AvailabilityService(session, settings)


def get_risk_service(session: DBSession, settings: Settings) -> RiskService:
    return RiskService(session, settings)


def get_commercial_service(session: DBSession, settings: Settings) -> CommercialService:
    return CommercialService(session, settings)


def get_rack_service(session: DBSession, settings: Settings) -> RackService:
    return RackService(session, settings)


def get_alert_service(session: DBSession) -> AlertQueryService:
    return AlertQueryService(session)


def get_market_service(session: DBSession, settings: Settings) -> MarketService:
    return MarketService(session, settings)


def get_health_service(session: DBSession) -> HealthService:
    return HealthService(session)


def get_monitoring_service(session: DBSession) -> MonitoringService:
    return MonitoringService(session)


def get_simulation_service(request: Request) -> SimulationService:
    return cast(SimulationService, request.app.state.simulation_service)


def get_copilot_service(request: Request) -> CopilotService:
    return cast(CopilotService, request.app.state.copilot_service)
