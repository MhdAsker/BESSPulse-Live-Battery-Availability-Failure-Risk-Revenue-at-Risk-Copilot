"""BESSPulse FastAPI application factory and production entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from api import __version__
from api.config import APISettings, get_settings
from api.errors import APIError
from api.routes import alerts, assets, copilot, health, market, racks, simulation
from api.schemas.common import ErrorResponse
from api.schemas.health import RootResponse, now_utc
from api.services import CopilotService, SimulationService

logger = logging.getLogger("besspulse.api")


def create_app(
    settings: APISettings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = None
        if session_factory is None:
            connect_args = (
                {"check_same_thread": False} if configured.database_url.startswith("sqlite") else {}
            )
            engine = create_engine(
                configured.database_url, connect_args=connect_args, pool_pre_ping=True
            )
            application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        else:
            application.state.session_factory = session_factory
        application.state.settings = configured
        application.state.simulation_service = SimulationService()
        application.state.copilot_service = CopilotService()
        yield
        if engine is not None:
            engine.dispose()

    application = FastAPI(
        title="BESSPulse API",
        version=__version__,
        description=(
            "Typed access to persisted BESS telemetry, availability, delivery risk, "
            "anomalies, alerts, market context, and counterfactual commercial analytics."
        ),
        contact={"name": "BESSPulse engineering"},
        license_info={"name": "Internal project"},
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness."},
            {"name": "assets", "description": "Asset state and analytical outputs."},
            {"name": "racks", "description": "Observable rack state and anomalies."},
            {"name": "alerts", "description": "Decision-support alert history."},
            {"name": "market", "description": "Observed and forecast market data."},
            {
                "name": "simulation controls (development/demo)",
                "description": "Configurable non-production controls.",
            },
            {
                "name": "copilot",
                "description": "Reserved contract; backend unavailable in Prompt 11.",
            },
        ],
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.api_cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
    )
    _register_middleware(application)
    _register_exception_handlers(application)
    for router in (
        health.router,
        assets.router,
        racks.router,
        alerts.router,
        market.router,
        simulation.router,
        copilot.router,
    ):
        application.include_router(router, prefix="/api/v1")

    @application.get("/", response_model=RootResponse, tags=["health"])
    def root() -> RootResponse:
        return RootResponse(version=__version__, timestamp_utc=now_utc())

    return application


def _register_middleware(application: FastAPI) -> None:
    @application.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if supplied and len(supplied) <= 128 else str(uuid4())
        request.state.request_id = request_id
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "api_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response


def _register_exception_handlers(application: FastAPI) -> None:
    def payload(request: Request, code: str, message: str, details: Any = None) -> dict[str, Any]:
        return ErrorResponse(
            error_code=code,
            message=message,
            details=details,
            request_id=getattr(request.state, "request_id", None),
            timestamp_utc=datetime.now(UTC),
        ).model_dump(mode="json")

    @application.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=payload(request, exc.error_code, exc.message, exc.details),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=payload(request, "validation_error", "Request validation failed.", details),
        )

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error(
            "database_error request_id=%s exception_type=%s",
            getattr(request.state, "request_id", None),
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=503,
            content=payload(request, "database_unavailable", "Database service is unavailable."),
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "internal_error request_id=%s exception_type=%s",
            getattr(request.state, "request_id", None),
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content=payload(request, "internal_error", "An internal server error occurred."),
        )


app = create_app()
