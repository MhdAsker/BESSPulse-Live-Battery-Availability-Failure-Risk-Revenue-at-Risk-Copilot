"""Rack-level deterministic technical, power, and energy capability."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from availability.config import AvailabilityConfig
from availability.schemas import ComponentState, RackCapability

REQUIRED_RACK_FIELDS = (
    "timestamp_utc",
    "rack_id",
    "pcs_id",
    "availability",
    "operating_state",
    "soc",
    "soh_proxy",
    "temperature_mean_c",
)


def thermal_derating_factor(temperature_c: float, config: AvailabilityConfig) -> float:
    thermal = config.simulation.thermal
    if temperature_c <= thermal.derating_start_c:
        return 1.0
    if temperature_c >= thermal.shutdown_temperature_c:
        return 0.0
    return (thermal.shutdown_temperature_c - temperature_c) / (
        thermal.shutdown_temperature_c - thermal.derating_start_c
    )


def _unknown_capability(
    timestamp: datetime, rack_id: str, pcs_id: str, config: AvailabilityConfig, reason: str
) -> RackCapability:
    return RackCapability(
        timestamp_utc=timestamp,
        component_id=rack_id,
        pcs_id=pcs_id,
        state=ComponentState.UNKNOWN,
        technical_available=None,
        soc=None,
        soh_proxy=None,
        thermal_derating_factor=0.0,
        usable_capacity_mwh=0.0,
        nominal_power_mw=config.nominal_rack_power_mw,
        available_discharge_power_mw=0.0,
        available_charge_power_mw=0.0,
        available_discharge_energy_mwh=0.0,
        available_charge_energy_mwh=0.0,
        limiting_factors=(reason,),
    )


def calculate_rack_availability(
    telemetry: Mapping[str, Any],
    config: AvailabilityConfig,
    *,
    reference_timestamp: datetime | pd.Timestamp | None = None,
    parent_pcs_available: bool | None = True,
) -> RackCapability:
    missing = [field for field in REQUIRED_RACK_FIELDS if field not in telemetry]
    fallback_time = pd.Timestamp(reference_timestamp or datetime.now(UTC)).to_pydatetime()
    rack_id = str(telemetry.get("rack_id", "UNKNOWN-RACK"))
    pcs_id = str(telemetry.get("pcs_id", "UNKNOWN-PCS"))
    if missing:
        return _unknown_capability(fallback_time, rack_id, pcs_id, config, "MISSING_TELEMETRY")
    timestamp = pd.Timestamp(telemetry["timestamp_utc"])
    if timestamp.tzinfo is None:
        raise ValueError("rack telemetry timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    required_values = [telemetry[field] for field in REQUIRED_RACK_FIELDS]
    if any(pd.isna(value) for value in required_values):
        return _unknown_capability(
            timestamp.to_pydatetime(), rack_id, pcs_id, config, "MISSING_TELEMETRY"
        )
    if reference_timestamp is not None:
        reference = pd.Timestamp(reference_timestamp)
        if reference.tzinfo is None:
            raise ValueError("reference timestamp must be timezone-aware")
        age = reference.tz_convert("UTC") - timestamp
        if age < pd.Timedelta(0):
            raise ValueError("future rack telemetry cannot be used")
        if age > pd.Timedelta(config.maximum_telemetry_staleness):
            return _unknown_capability(
                timestamp.to_pydatetime(), rack_id, pcs_id, config, "STALE_TELEMETRY"
            )
    if parent_pcs_available is None:
        return _unknown_capability(
            timestamp.to_pydatetime(), rack_id, pcs_id, config, "PCS_UNKNOWN"
        )
    observable_available = bool(telemetry["availability"])
    mode = str(telemetry["operating_state"]).upper()
    operational = observable_available and mode not in {"OFFLINE", "FAULTED"}
    if not parent_pcs_available:
        operational = False
    state = ComponentState.AVAILABLE if operational else ComponentState.UNAVAILABLE
    soc = float(telemetry["soc"])
    soh = float(telemetry["soh_proxy"])
    temperature = float(telemetry["temperature_mean_c"])
    if not all(np.isfinite(value) for value in (soc, soh, temperature)):
        return _unknown_capability(
            timestamp.to_pydatetime(), rack_id, pcs_id, config, "INVALID_TELEMETRY"
        )
    soc = min(1.0, max(0.0, soc))
    soh = min(1.0, max(0.0, soh))
    usable_capacity = config.nominal_rack_energy_mwh * soh
    discharge_energy = (
        usable_capacity
        * max(soc - config.minimum_soc, 0.0)
        * config.simulation.battery.discharge_efficiency
        if operational
        else 0.0
    )
    charge_energy = (
        usable_capacity
        * max(config.maximum_soc - soc, 0.0)
        / config.simulation.battery.charge_efficiency
        if operational
        else 0.0
    )
    thermal_factor = thermal_derating_factor(temperature, config) if operational else 0.0
    base_power = config.nominal_rack_power_mw * thermal_factor
    interval_hours = config.simulation.battery.telemetry_interval_minutes / 60
    discharge_power = min(base_power, discharge_energy / interval_hours)
    charge_power = min(base_power, charge_energy / interval_hours)
    factors: list[str] = []
    if not observable_available:
        factors.append("RACK_UNAVAILABLE")
    if not parent_pcs_available:
        factors.append("PCS_UNAVAILABLE")
    if thermal_factor < 1 and operational:
        factors.append("THERMAL_DERATING")
    if soc <= config.minimum_soc + config.availability_tolerance:
        factors.append("SOC_LOW")
    if soc >= config.maximum_soc - config.availability_tolerance:
        factors.append("SOC_HIGH")
    if soh < 1 - config.availability_tolerance:
        factors.append("CAPACITY_FADE")
    return RackCapability(
        timestamp_utc=timestamp.to_pydatetime(),
        component_id=rack_id,
        pcs_id=pcs_id,
        state=state,
        technical_available=operational,
        soc=soc,
        soh_proxy=soh,
        thermal_derating_factor=thermal_factor,
        usable_capacity_mwh=usable_capacity,
        nominal_power_mw=config.nominal_rack_power_mw,
        available_discharge_power_mw=min(config.nominal_rack_power_mw, discharge_power),
        available_charge_power_mw=min(config.nominal_rack_power_mw, charge_power),
        available_discharge_energy_mwh=max(0.0, discharge_energy),
        available_charge_energy_mwh=max(0.0, charge_energy),
        limiting_factors=tuple(dict.fromkeys(factors)) or ("NONE",),
    )
