"""Typed configuration for counterfactual battery dispatch optimization."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DispatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    healthy_power_mw: float = Field(default=20.0, gt=0)
    healthy_rated_energy_mwh: float = Field(default=40.0, gt=0)
    minimum_soc: float = Field(default=0.10, ge=0, lt=1)
    maximum_soc: float = Field(default=0.90, gt=0, le=1)
    initial_soc_fraction: float = Field(default=0.50, ge=0, le=1)
    terminal_policy: Literal["equal_initial_fraction", "minimum_initial_fraction"] = (
        "equal_initial_fraction"
    )
    terminal_tolerance_mwh: float = Field(default=1e-6, ge=0)
    healthy_charge_efficiency: float = Field(default=0.96, gt=0, le=1)
    healthy_discharge_efficiency: float = Field(default=0.96, gt=0, le=1)
    degradation_cost_eur_per_mwh: float = Field(default=2.0, ge=0)
    solver_name: str = "HIGHS"
    simultaneous_power_tolerance_mw: float = Field(default=1e-5, ge=0)
    feasibility_tolerance: float = Field(default=1e-5, gt=0)
    commercial_horizon_hours: int = Field(default=24, ge=1)
    capability_version: str = "availability_v1"
    optimization_version: str = "commercial_dispatch_v1"

    @property
    def healthy_usable_energy_mwh(self) -> float:
        return self.healthy_rated_energy_mwh * (self.maximum_soc - self.minimum_soc)
