"""Directional availability API contract."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, Freshness, Provenance


class AvailabilityResponse(APIModel):
    asset_id: str
    timestamp_utc: datetime
    technical_availability: float = Field(ge=0, le=1)
    rack_technical_availability: float = Field(ge=0, le=1)
    pcs_technical_availability: float = Field(ge=0, le=1)
    available_discharge_power_mw: float = Field(ge=0)
    available_charge_power_mw: float = Field(ge=0)
    discharge_power_availability: float = Field(ge=0, le=1)
    charge_power_availability: float = Field(ge=0, le=1)
    available_discharge_energy_mwh: float = Field(ge=0)
    available_charge_energy_mwh: float = Field(ge=0)
    discharge_energy_availability: float = Field(ge=0, le=1)
    charge_energy_availability: float = Field(ge=0, le=1)
    requested_power_availability: float | None = Field(default=None, ge=0, le=1)
    sustainable_request_duration_hours: float | None = Field(default=None, ge=0)
    limiting_factor: str
    limiting_factors: tuple[str, ...]
    limiting_components: tuple[str, ...]
    available_racks: int = Field(ge=0)
    available_pcs: int = Field(ge=0)
    freshness: Freshness
    data_provenance: Provenance
