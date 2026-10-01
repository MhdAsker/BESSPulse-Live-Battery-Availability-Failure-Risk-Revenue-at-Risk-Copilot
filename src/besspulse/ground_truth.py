"""Fault ground truth, isolated from feature-ready telemetry schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FaultType(StrEnum):
    NORMAL = "NORMAL"
    PCS_DERATING = "PCS_DERATING"
    RACK_OFFLINE = "RACK_OFFLINE"
    THERMAL_DRIFT = "THERMAL_DRIFT"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"
    SOC_SENSOR_BIAS = "SOC_SENSOR_BIAS"
    VOLTAGE_IMBALANCE = "VOLTAGE_IMBALANCE"
    EFFICIENCY_DEGRADATION = "EFFICIENCY_DEGRADATION"
    ACCELERATED_CAPACITY_FADE = "ACCELERATED_CAPACITY_FADE"
    SELF_DISCHARGE = "SELF_DISCHARGE"
    POWER_TRACKING_ERROR = "POWER_TRACKING_ERROR"


class FaultEvent(BaseModel):
    """Scheduled fault with half-open active interval [start, end)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fault_id: str
    fault_type: FaultType
    component_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    severity: float = Field(ge=0, le=1)
    progression_rate: float = Field(default=0, ge=0)
    ground_truth_label: str

    @field_validator("start_timestamp", "end_timestamp")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("fault timestamp must be timezone-aware")
        if offset.total_seconds() != 0:
            raise ValueError("fault timestamp must be UTC")
        return value

    @model_validator(mode="after")
    def end_after_start(self) -> "FaultEvent":
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("fault end_timestamp must be after start_timestamp")
        return self


class FaultGroundTruthRecord(BaseModel):
    """Point-in-time label stored separately from operational telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    fault_id: str
    fault_type: FaultType
    component_id: str
    severity: float
    ground_truth_label: str
