"""Typed availability policy layered on the central simulator configuration."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from besspulse.config import SimulationConfig


class AvailabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = "BESS-001"
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    minimum_operational_rack_fraction: float = Field(default=0.0, ge=0, le=1)
    availability_tolerance: float = Field(default=1e-6, gt=0)
    maximum_telemetry_staleness: str = "10min"
    requested_energy_duration_hours: float = Field(default=1.0, gt=0)
    energy_reporting_basis: str = "deliverable_or_acceptable_ac"
    energy_reserve_policy: Literal["configured_soc_bounds"] = "configured_soc_bounds"
    derating_handling: Literal["observable_temperature_plus_normalized_pcs"] = (
        "observable_temperature_plus_normalized_pcs"
    )
    unknown_component_policy: Literal["installed_denominator_zero_contribution"] = (
        "installed_denominator_zero_contribution"
    )

    @field_validator("maximum_telemetry_staleness")
    @classmethod
    def staleness_is_nonnegative(cls, value: str) -> str:
        if pd.Timedelta(value) < pd.Timedelta(0):
            raise ValueError("maximum telemetry staleness cannot be negative")
        return value

    @property
    def rated_site_power_mw(self) -> float:
        return self.simulation.battery.site_rated_power_mw

    @property
    def rated_site_energy_mwh(self) -> float:
        return self.simulation.battery.site_rated_energy_mwh

    @property
    def pcs_count(self) -> int:
        return self.simulation.battery.pcs_count

    @property
    def racks_per_pcs(self) -> int:
        return self.simulation.battery.racks_per_pcs

    @property
    def total_racks(self) -> int:
        return self.pcs_count * self.racks_per_pcs

    @property
    def nominal_rack_power_mw(self) -> float:
        return self.rated_site_power_mw / self.total_racks

    @property
    def nominal_rack_energy_mwh(self) -> float:
        return self.rated_site_energy_mwh / self.total_racks

    @property
    def nominal_pcs_power_mw(self) -> float:
        return self.rated_site_power_mw / self.pcs_count

    @property
    def minimum_soc(self) -> float:
        return self.simulation.battery.soc_min

    @property
    def maximum_soc(self) -> float:
        return self.simulation.battery.soc_max

    @property
    def minimum_request_threshold_mw(self) -> float:
        return self.simulation.minimum_request_threshold_mw

    @property
    def maximum_charge_power_mw(self) -> float:
        return self.rated_site_power_mw

    @property
    def maximum_discharge_power_mw(self) -> float:
        return self.rated_site_power_mw

    @property
    def rated_discharge_usable_energy_mwh(self) -> float:
        window = self.maximum_soc - self.minimum_soc
        return self.rated_site_energy_mwh * window * self.simulation.battery.discharge_efficiency

    @property
    def rated_charge_acceptable_energy_mwh(self) -> float:
        window = self.maximum_soc - self.minimum_soc
        return self.rated_site_energy_mwh * window / self.simulation.battery.charge_efficiency
