"""Typed availability states, component capabilities, and site snapshots."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ComponentState(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class CapabilityBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    component_id: str
    state: ComponentState
    technical_available: bool | None
    available_discharge_power_mw: float = Field(ge=0)
    available_charge_power_mw: float = Field(ge=0)
    available_discharge_energy_mwh: float = Field(ge=0)
    available_charge_energy_mwh: float = Field(ge=0)
    limiting_factors: tuple[str, ...]
    data_provenance: str = "DERIVED ENGINEERING ANALYTIC"


class RackCapability(CapabilityBase):
    pcs_id: str
    soc: float | None
    soh_proxy: float | None
    thermal_derating_factor: float = Field(ge=0, le=1)
    usable_capacity_mwh: float = Field(ge=0)
    nominal_power_mw: float = Field(gt=0)


class PCSCapability(CapabilityBase):
    available_rack_count: int = Field(ge=0)
    unknown_rack_count: int = Field(ge=0)
    total_rack_count: int = Field(gt=0)
    available_rack_fraction: float = Field(ge=0, le=1)
    pcs_derating_factor: float = Field(ge=0, le=1)
    nominal_power_mw: float = Field(gt=0)


class RequestedPowerAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_power_mw: float
    requested_power_is_active: bool
    requested_direction: str
    requested_power_availability: float | None
    requested_energy_availability: float | None
    sustainable_request_duration_hours: float | None
    available_power_in_requested_direction_mw: float | None
    available_energy_in_requested_direction_mwh: float | None


class AvailabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    asset_id: str
    rated_power_mw: float
    rated_energy_mwh: float
    technical_availability: float = Field(ge=0, le=1)
    rack_technical_availability: float = Field(ge=0, le=1)
    pcs_technical_availability: float = Field(ge=0, le=1)
    known_component_fraction: float = Field(ge=0, le=1)
    available_discharge_power_mw: float = Field(ge=0)
    available_charge_power_mw: float = Field(ge=0)
    discharge_power_availability: float = Field(ge=0, le=1)
    charge_power_availability: float = Field(ge=0, le=1)
    available_discharge_energy_mwh: float = Field(ge=0)
    available_charge_energy_mwh: float = Field(ge=0)
    discharge_energy_availability: float = Field(ge=0, le=1)
    charge_energy_availability: float = Field(ge=0, le=1)
    available_power_mw: float = Field(ge=0)
    available_energy_mwh: float = Field(ge=0)
    charge_headroom_mwh: float = Field(ge=0)
    power_availability: float = Field(ge=0, le=1)
    energy_availability: float = Field(ge=0, le=1)
    requested_power_mw: float
    requested_power_is_active: bool
    requested_direction: str
    requested_power_availability: float | None
    requested_energy_availability: float | None
    sustainable_request_duration_hours: float | None
    sustainable_discharge_power_15m_mw: float = Field(ge=0)
    sustainable_discharge_power_1h_mw: float = Field(ge=0)
    sustainable_discharge_power_2h_mw: float = Field(ge=0)
    sustainable_charge_power_15m_mw: float = Field(ge=0)
    sustainable_charge_power_1h_mw: float = Field(ge=0)
    sustainable_charge_power_2h_mw: float = Field(ge=0)
    actual_power_mw: float | None
    delivery_ratio: float | None
    observed_delivery_success: bool | None
    capability_delivery_gap: float | None
    available_racks: int = Field(ge=0)
    total_racks: int = Field(gt=0)
    unknown_racks: int = Field(ge=0)
    available_pcs: int = Field(ge=0)
    total_pcs: int = Field(gt=0)
    unknown_pcs: int = Field(ge=0)
    limiting_factor: str
    limiting_factors: tuple[str, ...]
    limiting_component_ids: tuple[str, ...]
    data_provenance: str = "DERIVED ENGINEERING ANALYTIC"
