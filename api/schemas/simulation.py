"""Development-only simulation control contracts."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from api.schemas.common import APIModel
from besspulse.ground_truth import FaultType


class FaultInjectionRequest(APIModel):
    fault_type: FaultType
    component_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Z0-9-]+$")
    start_timestamp: datetime | None = None
    duration_minutes: int = Field(default=60, ge=1, le=10080)
    severity: float = Field(ge=0, le=1)
    progression_rate: float = Field(default=0, ge=0, le=1)

    @field_validator("start_timestamp")
    @classmethod
    def timestamp_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("start_timestamp must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("start_timestamp must be UTC")
        return value


class FaultInjectionResponse(APIModel):
    fault_id: UUID
    fault_type: FaultType
    component_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    severity: float = Field(ge=0, le=1)
    status: str


class SimulationResetRequest(APIModel):
    asset_id: str = "BESS-001"
    seed: int | None = None
    reset_database: bool = False

    @model_validator(mode="after")
    def database_reset_not_supported(self) -> "SimulationResetRequest":
        if self.reset_database:
            raise ValueError("database reset is not supported by this endpoint")
        return self


class SimulationResetResponse(APIModel):
    asset_id: str
    seed: int
    status: str
    reset_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


def default_fault_window(duration_minutes: int) -> tuple[datetime, datetime]:
    start = datetime.now(UTC)
    return start, start + timedelta(minutes=duration_minutes)
