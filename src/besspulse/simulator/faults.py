"""Extensible deterministic fault-effect registry."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from besspulse.ground_truth import FaultEvent, FaultType


@dataclass
class FaultEffects:
    availability: bool = True
    power_factor: float = 1.0
    tracking_factor: float = 1.0
    temperature_offset_c: float = 0.0
    cooling_factor: float = 1.0
    soc_sensor_bias: float = 0.0
    voltage_spread_v: float = 2.0
    efficiency_factor: float = 1.0
    capacity_fade_factor: float = 1.0
    self_discharge_per_hour: float = 0.0


FaultHandler = Callable[[FaultEffects, float], None]


class FaultEngine:
    """Maps scheduled faults to physical/sensor effects without touching telemetry models."""

    def __init__(self, events: list[FaultEvent] | None = None) -> None:
        self.events = tuple(sorted(events or [], key=lambda event: event.start_timestamp))
        self._handlers: dict[FaultType, FaultHandler] = {
            FaultType.NORMAL: lambda effects, level: None,
            FaultType.PCS_DERATING: self._pcs_derating,
            FaultType.RACK_OFFLINE: self._rack_offline,
            FaultType.THERMAL_DRIFT: self._thermal_drift,
            FaultType.COOLING_DEGRADATION: self._cooling_degradation,
            FaultType.SOC_SENSOR_BIAS: self._soc_sensor_bias,
            FaultType.VOLTAGE_IMBALANCE: self._voltage_imbalance,
            FaultType.EFFICIENCY_DEGRADATION: self._efficiency_degradation,
            FaultType.ACCELERATED_CAPACITY_FADE: self._accelerated_capacity_fade,
            FaultType.SELF_DISCHARGE: self._self_discharge,
            FaultType.POWER_TRACKING_ERROR: self._power_tracking_error,
        }

    def register(self, fault_type: FaultType, handler: FaultHandler) -> None:
        """Allow a fault effect to be extended/replaced without changing simulator physics."""

        self._handlers[fault_type] = handler

    def active_events(self, timestamp: datetime) -> tuple[FaultEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.start_timestamp <= timestamp < event.end_timestamp
        )

    def effects_for(self, component_id: str, timestamp: datetime) -> FaultEffects:
        effects = FaultEffects()
        for event in self.active_events(timestamp):
            if event.component_id != component_id:
                continue
            elapsed_h = (timestamp - event.start_timestamp).total_seconds() / 3600
            level = min(1.0, event.severity + event.progression_rate * elapsed_h)
            self._handlers[event.fault_type](effects, level)
        return effects

    @staticmethod
    def _pcs_derating(effects: FaultEffects, level: float) -> None:
        effects.power_factor *= max(0.0, 1.0 - level)

    @staticmethod
    def _rack_offline(effects: FaultEffects, level: float) -> None:
        if level > 0:
            effects.availability = False
            effects.power_factor = 0.0

    @staticmethod
    def _thermal_drift(effects: FaultEffects, level: float) -> None:
        effects.temperature_offset_c += 15.0 * level

    @staticmethod
    def _cooling_degradation(effects: FaultEffects, level: float) -> None:
        effects.cooling_factor *= max(0.1, 1.0 - 0.9 * level)

    @staticmethod
    def _soc_sensor_bias(effects: FaultEffects, level: float) -> None:
        effects.soc_sensor_bias += 0.15 * level

    @staticmethod
    def _voltage_imbalance(effects: FaultEffects, level: float) -> None:
        effects.voltage_spread_v += 35.0 * level

    @staticmethod
    def _efficiency_degradation(effects: FaultEffects, level: float) -> None:
        effects.efficiency_factor *= max(0.7, 1.0 - 0.25 * level)

    @staticmethod
    def _accelerated_capacity_fade(effects: FaultEffects, level: float) -> None:
        effects.capacity_fade_factor *= 1.0 + 100.0 * level

    @staticmethod
    def _self_discharge(effects: FaultEffects, level: float) -> None:
        effects.self_discharge_per_hour += 0.005 * level

    @staticmethod
    def _power_tracking_error(effects: FaultEffects, level: float) -> None:
        effects.tracking_factor *= max(0.0, 1.0 - 0.5 * level)


def combine_effects(*items: FaultEffects) -> FaultEffects:
    """Compose independent site, PCS, and rack effects."""

    combined = FaultEffects()
    for item in items:
        combined.availability = combined.availability and item.availability
        combined.power_factor *= item.power_factor
        combined.tracking_factor *= item.tracking_factor
        combined.temperature_offset_c += item.temperature_offset_c
        combined.cooling_factor *= item.cooling_factor
        combined.soc_sensor_bias += item.soc_sensor_bias
        combined.voltage_spread_v += item.voltage_spread_v - 2.0
        combined.efficiency_factor *= item.efficiency_factor
        combined.capacity_fade_factor *= item.capacity_fade_factor
        combined.self_discharge_per_hour += item.self_discharge_per_hour
    return combined
