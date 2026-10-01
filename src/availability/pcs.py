"""PCS aggregation with rack conservation and converter constraints."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from availability.config import AvailabilityConfig
from availability.schemas import ComponentState, PCSCapability, RackCapability


def calculate_pcs_availability(
    pcs_id: str,
    timestamp_utc: datetime,
    rack_capabilities: Sequence[RackCapability],
    telemetry: Mapping[str, Any] | None,
    config: AvailabilityConfig,
) -> PCSCapability:
    expected_racks = config.racks_per_pcs
    available_racks = sum(item.state == ComponentState.AVAILABLE for item in rack_capabilities)
    unknown_racks = expected_racks - sum(
        item.state != ComponentState.UNKNOWN for item in rack_capabilities
    )
    rack_fraction = available_racks / expected_racks
    telemetry_known = (
        telemetry is not None
        and "availability" in telemetry
        and "derating_factor" in telemetry
        and pd.notna(telemetry["availability"])
        and pd.notna(telemetry["derating_factor"])
    )
    if not telemetry_known:
        state = ComponentState.UNKNOWN
        technical: bool | None = None
        inferred_derating = 0.0
    else:
        assert telemetry is not None
        minimum_met = (
            available_racks > 0
            and rack_fraction + config.availability_tolerance
            >= config.minimum_operational_rack_fraction
        )
        technical = bool(telemetry["availability"]) and minimum_met
        state = ComponentState.AVAILABLE if technical else ComponentState.UNAVAILABLE
        requested = float(telemetry.get("requested_power_mw", 0.0))
        rack_directional_power = sum(
            item.available_discharge_power_mw if requested >= 0 else item.available_charge_power_mw
            for item in rack_capabilities
        )
        expected_fraction = min(1.0, rack_directional_power / config.nominal_pcs_power_mw)
        observed_factor = min(1.0, max(0.0, float(telemetry["derating_factor"])))
        inferred_derating = (
            min(1.0, observed_factor / expected_fraction)
            if expected_fraction > config.availability_tolerance
            else 1.0
        )
    rack_discharge_power = sum(item.available_discharge_power_mw for item in rack_capabilities)
    rack_charge_power = sum(item.available_charge_power_mw for item in rack_capabilities)
    converter_limit = config.nominal_pcs_power_mw * inferred_derating
    if state != ComponentState.AVAILABLE:
        discharge_power = 0.0
        charge_power = 0.0
        discharge_energy = 0.0
        charge_energy = 0.0
    else:
        discharge_power = min(rack_discharge_power, converter_limit)
        charge_power = min(rack_charge_power, converter_limit)
        discharge_energy = sum(item.available_discharge_energy_mwh for item in rack_capabilities)
        charge_energy = sum(item.available_charge_energy_mwh for item in rack_capabilities)
    factors = {
        factor for item in rack_capabilities for factor in item.limiting_factors if factor != "NONE"
    }
    if state == ComponentState.UNKNOWN:
        factors.add("PCS_UNKNOWN")
    elif state == ComponentState.UNAVAILABLE:
        factors.add("PCS_UNAVAILABLE")
    if inferred_derating < 1 - config.availability_tolerance and state == ComponentState.AVAILABLE:
        factors.add("POWER_LIMIT")
    return PCSCapability(
        timestamp_utc=timestamp_utc,
        component_id=pcs_id,
        state=state,
        technical_available=technical,
        available_rack_count=available_racks,
        unknown_rack_count=max(0, unknown_racks),
        total_rack_count=expected_racks,
        available_rack_fraction=rack_fraction,
        pcs_derating_factor=inferred_derating,
        nominal_power_mw=config.nominal_pcs_power_mw,
        available_discharge_power_mw=max(0.0, discharge_power),
        available_charge_power_mw=max(0.0, charge_power),
        available_discharge_energy_mwh=max(0.0, discharge_energy),
        available_charge_energy_mwh=max(0.0, charge_energy),
        limiting_factors=tuple(sorted(factors)) or ("NONE",),
    )
