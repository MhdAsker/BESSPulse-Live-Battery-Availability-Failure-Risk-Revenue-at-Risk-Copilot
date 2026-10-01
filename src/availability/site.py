"""Site-level technical, power, energy, and request-context aggregation."""

from collections.abc import Sequence
from datetime import datetime

from availability.config import AvailabilityConfig
from availability.requested_power import (
    calculate_requested_power_availability,
    sustainable_power,
)
from availability.schemas import (
    AvailabilitySnapshot,
    ComponentState,
    PCSCapability,
    RackCapability,
)


def _fraction(numerator: float, denominator: float) -> float:
    return min(1.0, max(0.0, numerator / denominator)) if denominator > 0 else 0.0


def _duration_power(power: float, energy: float, hours: float) -> float:
    return sustainable_power(power, energy, hours)


def calculate_site_availability(
    timestamp_utc: datetime,
    rack_capabilities: Sequence[RackCapability],
    pcs_capabilities: Sequence[PCSCapability],
    requested_power_mw: float,
    actual_power_mw: float | None,
    config: AvailabilityConfig,
) -> AvailabilitySnapshot:
    available_racks = sum(item.state == ComponentState.AVAILABLE for item in rack_capabilities)
    unknown_racks = config.total_racks - sum(
        item.state != ComponentState.UNKNOWN for item in rack_capabilities
    )
    available_pcs = sum(item.state == ComponentState.AVAILABLE for item in pcs_capabilities)
    unknown_pcs = config.pcs_count - sum(
        item.state != ComponentState.UNKNOWN for item in pcs_capabilities
    )
    rack_technical = available_racks / config.total_racks
    pcs_technical = available_pcs / config.pcs_count
    technical = rack_technical
    known = (config.total_racks + config.pcs_count - unknown_racks - unknown_pcs) / (
        config.total_racks + config.pcs_count
    )
    discharge_power = min(
        config.maximum_discharge_power_mw,
        sum(item.available_discharge_power_mw for item in pcs_capabilities),
    )
    charge_power = min(
        config.maximum_charge_power_mw,
        sum(item.available_charge_power_mw for item in pcs_capabilities),
    )
    discharge_energy = min(
        config.rated_discharge_usable_energy_mwh,
        sum(item.available_discharge_energy_mwh for item in pcs_capabilities),
    )
    charge_energy = min(
        config.rated_charge_acceptable_energy_mwh,
        sum(item.available_charge_energy_mwh for item in pcs_capabilities),
    )
    request = calculate_requested_power_availability(
        requested_power_mw,
        available_discharge_power_mw=discharge_power,
        available_charge_power_mw=charge_power,
        available_discharge_energy_mwh=discharge_energy,
        available_charge_energy_mwh=charge_energy,
        config=config,
    )
    discharge_power_availability = _fraction(discharge_power, config.rated_site_power_mw)
    charge_power_availability = _fraction(charge_power, config.rated_site_power_mw)
    discharge_energy_availability = _fraction(
        discharge_energy, config.rated_discharge_usable_energy_mwh
    )
    charge_energy_availability = _fraction(charge_energy, config.rated_charge_acceptable_energy_mwh)
    if request.requested_direction == "DISCHARGE":
        contextual_power = discharge_power
        contextual_power_availability = discharge_power_availability
    elif request.requested_direction == "CHARGE":
        contextual_power = charge_power
        contextual_power_availability = charge_power_availability
    else:
        contextual_power = min(discharge_power, charge_power)
        contextual_power_availability = min(discharge_power_availability, charge_power_availability)
    if request.requested_power_is_active and actual_power_mw is not None:
        delivery_ratio = abs(actual_power_mw) / abs(requested_power_mw)
        delivery_success: bool | None = delivery_ratio >= config.simulation.delivery_failure_ratio
        capability_gap = (
            delivery_ratio - request.requested_power_availability
            if request.requested_power_availability is not None
            else None
        )
    else:
        delivery_ratio = None
        delivery_success = None
        capability_gap = None
    factors = {
        factor
        for item in [*rack_capabilities, *pcs_capabilities]
        for factor in item.limiting_factors
        if factor != "NONE"
    }
    if discharge_power_availability < 1 - config.availability_tolerance:
        factors.add("POWER_LIMIT")
    if discharge_energy_availability < 1 - config.availability_tolerance:
        factors.add("ENERGY_LIMIT")
    ordered_factors = tuple(sorted(factors)) or ("NONE",)
    limiting_ids = tuple(
        sorted(
            item.component_id
            for item in [*rack_capabilities, *pcs_capabilities]
            if item.limiting_factors != ("NONE",) or item.state != ComponentState.AVAILABLE
        )
    )
    return AvailabilitySnapshot(
        timestamp_utc=timestamp_utc,
        asset_id=config.asset_id,
        rated_power_mw=config.rated_site_power_mw,
        rated_energy_mwh=config.rated_site_energy_mwh,
        technical_availability=technical,
        rack_technical_availability=rack_technical,
        pcs_technical_availability=pcs_technical,
        known_component_fraction=known,
        available_discharge_power_mw=discharge_power,
        available_charge_power_mw=charge_power,
        discharge_power_availability=discharge_power_availability,
        charge_power_availability=charge_power_availability,
        available_discharge_energy_mwh=discharge_energy,
        available_charge_energy_mwh=charge_energy,
        discharge_energy_availability=discharge_energy_availability,
        charge_energy_availability=charge_energy_availability,
        available_power_mw=contextual_power,
        available_energy_mwh=discharge_energy,
        charge_headroom_mwh=charge_energy,
        power_availability=contextual_power_availability,
        energy_availability=discharge_energy_availability,
        requested_power_mw=requested_power_mw,
        requested_power_is_active=request.requested_power_is_active,
        requested_direction=request.requested_direction,
        requested_power_availability=request.requested_power_availability,
        requested_energy_availability=request.requested_energy_availability,
        sustainable_request_duration_hours=request.sustainable_request_duration_hours,
        sustainable_discharge_power_15m_mw=_duration_power(discharge_power, discharge_energy, 0.25),
        sustainable_discharge_power_1h_mw=_duration_power(discharge_power, discharge_energy, 1.0),
        sustainable_discharge_power_2h_mw=_duration_power(discharge_power, discharge_energy, 2.0),
        sustainable_charge_power_15m_mw=_duration_power(charge_power, charge_energy, 0.25),
        sustainable_charge_power_1h_mw=_duration_power(charge_power, charge_energy, 1.0),
        sustainable_charge_power_2h_mw=_duration_power(charge_power, charge_energy, 2.0),
        actual_power_mw=actual_power_mw,
        delivery_ratio=delivery_ratio,
        observed_delivery_success=delivery_success,
        capability_delivery_gap=capability_gap,
        available_racks=available_racks,
        total_racks=config.total_racks,
        unknown_racks=max(0, unknown_racks),
        available_pcs=available_pcs,
        total_pcs=config.pcs_count,
        unknown_pcs=max(0, unknown_pcs),
        limiting_factor=ordered_factors[0] if len(ordered_factors) == 1 else "MULTIPLE",
        limiting_factors=ordered_factors,
        limiting_component_ids=limiting_ids,
    )
