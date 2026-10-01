"""Rack state and reduced-order physics."""

from dataclasses import dataclass
from datetime import datetime

from besspulse.config import SimulationConfig
from besspulse.schemas import OperatingMode, RackTelemetry
from besspulse.simulator.faults import FaultEffects


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass
class RackState:
    rack_id: str
    pcs_id: str
    true_soc: float
    soh_proxy: float = 1.0
    temperature_c: float = 20.0
    cumulative_throughput_mwh: float = 0.0
    charge_energy_mwh: float = 0.0
    discharge_energy_mwh: float = 0.0
    previous_power_fraction: float = 0.0


class Rack:
    def __init__(self, state: RackState, config: SimulationConfig) -> None:
        self.state = state
        self.config = config
        count = config.battery.pcs_count * config.battery.racks_per_pcs
        self.nominal_capacity_mwh = config.battery.site_rated_energy_mwh / count
        self.rated_power_mw = config.battery.site_rated_power_mw / count

    def power_limit(self, effects: FaultEffects, dt_hours: float) -> tuple[float, float]:
        """Return available discharge and charge magnitudes in MW."""

        if not effects.availability:
            return 0.0, 0.0
        cfg = self.config.battery
        thermal = self._thermal_derating(self.state.temperature_c + effects.temperature_offset_c)
        nameplate = self.rated_power_mw * effects.power_factor * thermal
        usable_capacity = self.nominal_capacity_mwh * self.state.soh_proxy
        discharge_energy = max(0.0, self.state.true_soc - cfg.soc_min) * usable_capacity
        charge_room = max(0.0, cfg.soc_max - self.state.true_soc) * usable_capacity
        discharge = min(nameplate, discharge_energy * cfg.discharge_efficiency / dt_hours)
        charge = min(nameplate, charge_room / (cfg.charge_efficiency * dt_hours))
        return discharge, charge

    def step(
        self,
        timestamp: datetime,
        requested_power_mw: float,
        effects: FaultEffects,
        ambient_temperature_c: float,
        dt_hours: float,
    ) -> RackTelemetry:
        cfg = self.config.battery
        discharge_limit, charge_limit = self.power_limit(effects, dt_hours)
        actual = _clamp(requested_power_mw, -charge_limit, discharge_limit)
        actual *= effects.tracking_factor
        efficiency = self._efficiency(actual, effects)
        capacity = self.nominal_capacity_mwh * self.state.soh_proxy

        if actual >= 0:
            internal_delta_mwh = -(actual * dt_hours / max(efficiency, 1e-9))
            self.state.discharge_energy_mwh += actual * dt_hours
        else:
            internal_delta_mwh = -actual * dt_hours * efficiency
            self.state.charge_energy_mwh += -actual * dt_hours
        internal_delta_mwh -= effects.self_discharge_per_hour * capacity * dt_hours
        self.state.true_soc = _clamp(
            self.state.true_soc + internal_delta_mwh / max(capacity, 1e-9),
            cfg.soc_min,
            cfg.soc_max,
        )
        interval_throughput = abs(actual) * dt_hours
        self.state.cumulative_throughput_mwh += interval_throughput
        self._update_temperature(actual, ambient_temperature_c, effects, dt_hours)
        self._update_degradation(interval_throughput, effects, dt_hours)

        voltage = cfg.nominal_voltage_v * (0.90 + 0.20 * self.state.true_soc)
        current_a = 0.0 if voltage == 0 else actual * 1_000_000 / voltage
        observed_soc = _clamp(self.state.true_soc + effects.soc_sensor_bias, 0.0, 1.0)
        observed_temperature = self.state.temperature_c + effects.temperature_offset_c
        temp_spread = 1.0 + 0.08 * abs(actual / max(self.rated_power_mw, 1e-9))
        available = effects.availability
        mode = (
            OperatingMode.DISCHARGING
            if actual > 1e-9
            else OperatingMode.CHARGING
            if actual < -1e-9
            else OperatingMode.IDLE
        )
        return RackTelemetry(
            timestamp_utc=timestamp,
            rack_id=self.state.rack_id,
            pcs_id=self.state.pcs_id,
            soc=observed_soc,
            soh_proxy=self.state.soh_proxy,
            voltage_v=voltage,
            current_a=current_a,
            temperature_mean_c=observed_temperature,
            temperature_min_c=observed_temperature - temp_spread / 2,
            temperature_max_c=observed_temperature + temp_spread / 2,
            temperature_spread_c=temp_spread,
            voltage_spread_v=effects.voltage_spread_v,
            requested_power_mw=requested_power_mw,
            actual_power_mw=actual,
            charge_energy_mwh=self.state.charge_energy_mwh,
            discharge_energy_mwh=self.state.discharge_energy_mwh,
            cumulative_throughput_mwh=self.state.cumulative_throughput_mwh,
            equivalent_full_cycles=self.state.cumulative_throughput_mwh
            / (2 * self.nominal_capacity_mwh),
            rte=_clamp(efficiency**2, 0.0, 1.0),
            availability=available,
            alarm_code=(
                None
                if available and observed_temperature < self.config.thermal.derating_start_c
                else "RACK_ALARM"
            ),
            operating_state=mode,
        )

    def _efficiency(self, power_mw: float, effects: FaultEffects) -> float:
        cfg = self.config.battery
        base = cfg.discharge_efficiency if power_mw >= 0 else cfg.charge_efficiency
        load = abs(power_mw) / max(self.rated_power_mw, 1e-9)
        temp_penalty = 0.0005 * abs(self.state.temperature_c - 25.0)
        soc_penalty = 0.015 * abs(self.state.true_soc - 0.5)
        load_penalty = 0.015 * (load - 0.5) ** 2
        ageing_penalty = 0.05 * (1.0 - self.state.soh_proxy)
        return _clamp(
            (base - temp_penalty - soc_penalty - load_penalty - ageing_penalty)
            * effects.efficiency_factor,
            0.70,
            0.99,
        )

    def _thermal_derating(self, temperature_c: float) -> float:
        thermal = self.config.thermal
        if temperature_c <= thermal.derating_start_c:
            return 1.0
        if temperature_c >= thermal.shutdown_temperature_c:
            return 0.0
        span = thermal.shutdown_temperature_c - thermal.derating_start_c
        return (thermal.shutdown_temperature_c - temperature_c) / span

    def _update_temperature(
        self,
        power_mw: float,
        ambient_c: float,
        effects: FaultEffects,
        dt_hours: float,
    ) -> None:
        thermal = self.config.thermal
        fraction = abs(power_mw) / max(self.rated_power_mw, 1e-9)
        effective_fraction = (
            1.0 - thermal.recent_power_weight
        ) * fraction + thermal.recent_power_weight * self.state.previous_power_fraction
        cooling = thermal.cooling_effectiveness * effects.cooling_factor
        relaxation = (
            (ambient_c - self.state.temperature_c) * cooling / thermal.thermal_time_constant_hours
        )
        heating = thermal.power_heating_c_per_hour * effective_fraction**2
        soc_heating = 0.2 * abs(self.state.true_soc - 0.5)
        self.state.temperature_c += (relaxation + heating + soc_heating) * dt_hours
        self.state.previous_power_fraction = fraction

    def _update_degradation(
        self, throughput_mwh: float, effects: FaultEffects, dt_hours: float
    ) -> None:
        deg = self.config.degradation
        efc_increment = throughput_mwh / (2 * self.nominal_capacity_mwh)
        hot_exposure = max(0.0, self.state.temperature_c - 30.0) / 10.0
        depth = abs(self.state.true_soc - 0.5) * 2.0
        fade = (
            deg.calendar_fade_per_day * dt_hours / 24
            + deg.throughput_fade_per_efc * efc_increment
            + deg.hot_temperature_fade_per_hour * hot_exposure * dt_hours
            + deg.depth_of_discharge_fade_per_unit_hour * depth * dt_hours
        ) * effects.capacity_fade_factor
        self.state.soh_proxy = _clamp(self.state.soh_proxy - fade, 0.5, 1.0)
