"""Validated, interval-specific physical capability contracts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimization.config import DispatchConfig


@dataclass(frozen=True)
class DispatchCapability:
    charge_power_mw: NDArray[np.float64]
    discharge_power_mw: NDArray[np.float64]
    usable_energy_capacity_mwh: NDArray[np.float64]
    available_charge_energy_mwh: NDArray[np.float64]
    available_discharge_energy_mwh: NDArray[np.float64]
    charge_efficiency: NDArray[np.float64]
    discharge_efficiency: NDArray[np.float64]
    technical_availability: NDArray[np.float64]
    name: str

    def validate(self, intervals: int) -> None:
        fields = {
            "charge_power_mw": self.charge_power_mw,
            "discharge_power_mw": self.discharge_power_mw,
            "usable_energy_capacity_mwh": self.usable_energy_capacity_mwh,
            "available_charge_energy_mwh": self.available_charge_energy_mwh,
            "available_discharge_energy_mwh": self.available_discharge_energy_mwh,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "technical_availability": self.technical_availability,
        }
        for name, values in fields.items():
            if values.shape != (intervals,):
                raise ValueError(f"{name} must have one value per market interval")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in [
            "charge_power_mw",
            "discharge_power_mw",
            "usable_energy_capacity_mwh",
            "available_charge_energy_mwh",
            "available_discharge_energy_mwh",
        ]:
            if (fields[name] < 0).any():
                raise ValueError(f"{name} cannot be negative")
        for name in ["charge_efficiency", "discharge_efficiency"]:
            if ((fields[name] <= 0) | (fields[name] > 1)).any():
                raise ValueError(f"{name} must be in (0, 1]")
        if ((self.technical_availability < 0) | (self.technical_availability > 1)).any():
            raise ValueError("technical_availability must be in [0, 1]")


def healthy_capability(intervals: int, config: DispatchConfig) -> DispatchCapability:
    capacity = config.healthy_usable_energy_mwh
    ones = np.ones(intervals, dtype=float)
    return DispatchCapability(
        charge_power_mw=ones * config.healthy_power_mw,
        discharge_power_mw=ones * config.healthy_power_mw,
        usable_energy_capacity_mwh=ones * capacity,
        available_charge_energy_mwh=ones * capacity / config.healthy_charge_efficiency,
        available_discharge_energy_mwh=ones * capacity * config.healthy_discharge_efficiency,
        charge_efficiency=ones * config.healthy_charge_efficiency,
        discharge_efficiency=ones * config.healthy_discharge_efficiency,
        technical_availability=ones,
        name="healthy_reference",
    )
