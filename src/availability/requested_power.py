"""Direction-aware requested-power and energy sufficiency assessment."""

from availability.config import AvailabilityConfig
from availability.schemas import RequestedPowerAssessment


def power_direction(requested_power_mw: float, threshold_mw: float) -> str:
    if requested_power_mw > threshold_mw:
        return "DISCHARGE"
    if requested_power_mw < -threshold_mw:
        return "CHARGE"
    return "IDLE"


def calculate_requested_power_availability(
    requested_power_mw: float,
    *,
    available_discharge_power_mw: float,
    available_charge_power_mw: float,
    available_discharge_energy_mwh: float,
    available_charge_energy_mwh: float,
    config: AvailabilityConfig,
) -> RequestedPowerAssessment:
    direction = power_direction(requested_power_mw, config.minimum_request_threshold_mw)
    if direction == "IDLE":
        return RequestedPowerAssessment(
            requested_power_mw=requested_power_mw,
            requested_power_is_active=False,
            requested_direction=direction,
            requested_power_availability=None,
            requested_energy_availability=None,
            sustainable_request_duration_hours=None,
            available_power_in_requested_direction_mw=None,
            available_energy_in_requested_direction_mwh=None,
        )
    requested_magnitude = abs(requested_power_mw)
    if direction == "DISCHARGE":
        available_power = available_discharge_power_mw
        available_energy = available_discharge_energy_mwh
    else:
        available_power = available_charge_power_mw
        available_energy = available_charge_energy_mwh
    power_ratio = min(available_power / requested_magnitude, 1.0)
    required_energy = requested_magnitude * config.requested_energy_duration_hours
    energy_ratio = min(available_energy / required_energy, 1.0)
    duration = available_energy / requested_magnitude
    return RequestedPowerAssessment(
        requested_power_mw=requested_power_mw,
        requested_power_is_active=True,
        requested_direction=direction,
        requested_power_availability=power_ratio,
        requested_energy_availability=energy_ratio,
        sustainable_request_duration_hours=duration,
        available_power_in_requested_direction_mw=available_power,
        available_energy_in_requested_direction_mwh=available_energy,
    )


def sustainable_power(power_limit_mw: float, available_energy_mwh: float, hours: float) -> float:
    if hours <= 0:
        raise ValueError("duration must be positive")
    return max(0.0, min(power_limit_mw, available_energy_mwh / hours))
