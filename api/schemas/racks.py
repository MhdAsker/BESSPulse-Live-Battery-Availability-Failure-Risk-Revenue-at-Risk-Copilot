"""Observable rack state response contract."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, Freshness, Provenance


class AnomalySummary(APIModel):
    active: bool
    peak_score: float | None = Field(default=None, ge=0)
    detector: str | None = None


class RackResponse(APIModel):
    rack_id: str
    pcs_id: str
    timestamp_utc: datetime
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
    rte: float = Field(ge=0, le=1)
    availability: bool
    alarm_code: str | None
    operating_state: str
    peer_temperature_mean_c: float | None = None
    peer_voltage_mean_v: float | None = None
    expected_temperature_c: float | None = None
    thermal_residual_c: float | None = None
    anomaly_summary: AnomalySummary
    freshness: Freshness
    data_provenance: Provenance
