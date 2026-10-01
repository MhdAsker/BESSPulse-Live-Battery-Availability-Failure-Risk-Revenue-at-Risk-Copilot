"""Public telemetry schemas; fault truth deliberately lives elsewhere."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Provenance = Literal["SIMULATED"]


class OperatingMode(StrEnum):
    CHARGING = "CHARGING"
    DISCHARGING = "DISCHARGING"
    IDLE = "IDLE"


class TelemetryBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    data_provenance: Provenance = "SIMULATED"

    @field_validator("timestamp_utc")
    @classmethod
    def timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("timestamp must be timezone-aware")
        if offset.total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return value

    def to_feature_dict(self) -> dict[str, Any]:
        """Return observables only; no ground-truth object is reachable here."""

        return self.model_dump()


class SiteTelemetry(TelemetryBase):
    site_requested_power_mw: float
    site_actual_power_mw: float
    available_power_mw: float = Field(ge=0)
    available_energy_mwh: float = Field(ge=0)
    site_soc: float = Field(ge=0, le=1)
    rte: float = Field(ge=0, le=1)
    ambient_temperature_c: float
    available_racks: int = Field(ge=0)
    available_pcs: int = Field(ge=0)
    operating_mode: OperatingMode
    alarm_count: int = Field(ge=0)


class PCSTelemetry(TelemetryBase):
    pcs_id: str
    requested_power_mw: float
    actual_power_mw: float
    efficiency: float = Field(ge=0, le=1)
    temperature_c: float
    availability: bool
    derating_factor: float = Field(ge=0, le=1)
    alarm_state: bool


class RackTelemetry(TelemetryBase):
    rack_id: str
    pcs_id: str
    soc: float = Field(ge=0, le=1)
    soh_proxy: float = Field(ge=0, le=1)
    voltage_v: float = Field(gt=0)
    current_a: float
    temperature_mean_c: float
    temperature_min_c: float
    temperature_max_c: float
    temperature_spread_c: float = Field(ge=0)
    voltage_spread_v: float = Field(ge=0)
    requested_power_mw: float
    actual_power_mw: float
    charge_energy_mwh: float = Field(ge=0)
    discharge_energy_mwh: float = Field(ge=0)
    cumulative_throughput_mwh: float = Field(ge=0)
    equivalent_full_cycles: float = Field(ge=0)
    rte: float = Field(ge=0, le=1)
    availability: bool
    alarm_code: str | None
    operating_state: OperatingMode
