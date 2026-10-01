"""Health and service metadata contracts."""

from datetime import UTC, datetime

from api.schemas.common import APIModel


class HealthResponse(APIModel):
    status: str
    service: str = "BESSPulse API"
    version: str
    liveness: str
    readiness: str
    database_status: str
    model_status: str
    market_data_status: str
    components: dict[str, str]
    timestamp_utc: datetime


class RootResponse(APIModel):
    service: str = "BESSPulse API"
    version: str
    api_base_path: str = "/api/v1"
    docs_url: str = "/docs"
    timestamp_utc: datetime


def now_utc() -> datetime:
    return datetime.now(UTC)
