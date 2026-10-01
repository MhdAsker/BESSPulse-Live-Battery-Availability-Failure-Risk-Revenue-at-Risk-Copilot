"""Site-level simulation orchestration."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from besspulse.config import SimulationConfig
from besspulse.ground_truth import FaultEvent, FaultGroundTruthRecord
from besspulse.schemas import OperatingMode, PCSTelemetry, RackTelemetry, SiteTelemetry
from besspulse.simulator.faults import FaultEffects, FaultEngine, combine_effects
from besspulse.simulator.pcs import PCS
from besspulse.simulator.rack import Rack, RackState


@dataclass(frozen=True)
class SimulationRun:
    """Outputs with operational telemetry and labels in separate collections."""

    site_telemetry: tuple[SiteTelemetry, ...]
    pcs_telemetry: tuple[PCSTelemetry, ...]
    rack_telemetry: tuple[RackTelemetry, ...]
    fault_ground_truth: tuple[FaultGroundTruthRecord, ...]


class BatterySimulator:
    """Configurable site → PCS → rack reduced-order digital twin."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        faults: list[FaultEvent] | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.simulation_seed)
        self.fault_engine = FaultEngine(faults)
        self.pcs_units = self._build_hierarchy()

    @property
    def racks(self) -> tuple[Rack, ...]:
        return tuple(rack for pcs in self.pcs_units for rack in pcs.racks)

    def _build_hierarchy(self) -> list[PCS]:
        result: list[PCS] = []
        cfg = self.config
        for pcs_index in range(1, cfg.battery.pcs_count + 1):
            pcs_id = f"PCS-{pcs_index:02d}"
            racks = [
                Rack(
                    RackState(
                        rack_id=f"{pcs_id}-RACK-{rack_index:02d}",
                        pcs_id=pcs_id,
                        true_soc=cfg.battery.initial_soc,
                        temperature_c=cfg.thermal.ambient_temperature_c,
                    ),
                    cfg,
                )
                for rack_index in range(1, cfg.battery.racks_per_pcs + 1)
            ]
            result.append(PCS(pcs_id=pcs_id, racks=racks))
        return result

    def simulate(
        self,
        start_timestamp: datetime,
        requested_power_mw: Iterable[float],
        ambient_temperature_c: float | Iterable[float] | None = None,
    ) -> SimulationRun:
        """Generate one telemetry row per supplied request, immediately/offline."""

        start = self._require_utc(start_timestamp)
        requests = list(requested_power_mw)
        ambients = self._ambient_series(ambient_temperature_c, len(requests))
        interval = timedelta(minutes=self.config.battery.telemetry_interval_minutes)
        site_rows: list[SiteTelemetry] = []
        pcs_rows: list[PCSTelemetry] = []
        rack_rows: list[RackTelemetry] = []
        truth_rows: list[FaultGroundTruthRecord] = []
        for index, (request, ambient) in enumerate(zip(requests, ambients, strict=True)):
            timestamp = start + index * interval
            site, pcs_step, rack_step = self._step(timestamp, request, ambient)
            site_rows.append(site)
            pcs_rows.extend(pcs_step)
            rack_rows.extend(rack_step)
            truth_rows.extend(self._ground_truth_at(timestamp))
        return SimulationRun(tuple(site_rows), tuple(pcs_rows), tuple(rack_rows), tuple(truth_rows))

    def _step(
        self, timestamp: datetime, request: float, ambient_c: float
    ) -> tuple[SiteTelemetry, list[PCSTelemetry], list[RackTelemetry]]:
        cfg = self.config
        dt_hours = cfg.battery.telemetry_interval_minutes / 60
        request = min(
            cfg.battery.site_rated_power_mw,
            max(-cfg.battery.site_rated_power_mw, request),
        )
        site_effects = self.fault_engine.effects_for("SITE", timestamp)

        contexts: list[tuple[PCS, FaultEffects, list[tuple[Rack, FaultEffects, float, float]]]] = []
        for pcs in self.pcs_units:
            pcs_effects = combine_effects(
                site_effects, self.fault_engine.effects_for(pcs.pcs_id, timestamp)
            )
            rack_context: list[tuple[Rack, FaultEffects, float, float]] = []
            for rack in pcs.racks:
                effects = combine_effects(
                    pcs_effects,
                    self.fault_engine.effects_for(rack.state.rack_id, timestamp),
                )
                discharge, charge = rack.power_limit(effects, dt_hours)
                rack_context.append((rack, effects, discharge, charge))
            contexts.append((pcs, pcs_effects, rack_context))

        use_discharge = request >= 0
        pcs_limits = [
            sum(item[2] if use_discharge else item[3] for item in rack_context)
            for _, _, rack_context in contexts
        ]
        total_limit = sum(pcs_limits)
        target = min(abs(request), total_limit)
        rack_output: list[RackTelemetry] = []
        pcs_output: list[PCSTelemetry] = []
        for (pcs, _pcs_effects, rack_context), pcs_limit in zip(contexts, pcs_limits, strict=True):
            pcs_target_mag = target * pcs_limit / total_limit if total_limit > 0 else 0.0
            pcs_request = pcs_target_mag if use_discharge else -pcs_target_mag
            rows: list[RackTelemetry] = []
            for rack, effects, discharge, charge in rack_context:
                rack_limit = discharge if use_discharge else charge
                rack_mag = pcs_target_mag * rack_limit / pcs_limit if pcs_limit > 0 else 0.0
                rack_request = rack_mag if use_discharge else -rack_mag
                row = rack.step(timestamp, rack_request, effects, ambient_c, dt_hours)
                rows.append(row)
                rack_output.append(row)
            pcs_nameplate = cfg.battery.site_rated_power_mw / cfg.battery.pcs_count
            derating = min(1.0, pcs_limit / pcs_nameplate)
            pcs_output.append(pcs.telemetry(timestamp, pcs_request, rows, derating))

        actual = sum(row.actual_power_mw for row in rack_output)
        available_racks = sum(row.availability for row in rack_output)
        available_pcs = sum(row.availability for row in pcs_output)
        alarms = sum(row.alarm_code is not None for row in rack_output)
        available_energy = sum(
            max(0.0, rack.state.true_soc - cfg.battery.soc_min)
            * rack.nominal_capacity_mwh
            * rack.state.soh_proxy
            for rack in self.racks
        )
        site_soc = sum(row.soc for row in rack_output) / len(rack_output)
        rte = sum(row.rte for row in rack_output) / len(rack_output)
        mode = (
            OperatingMode.DISCHARGING
            if actual > 1e-9
            else OperatingMode.CHARGING
            if actual < -1e-9
            else OperatingMode.IDLE
        )
        site_row = SiteTelemetry(
            timestamp_utc=timestamp,
            site_requested_power_mw=request,
            site_actual_power_mw=actual,
            available_power_mw=total_limit,
            available_energy_mwh=available_energy,
            site_soc=site_soc,
            rte=rte,
            ambient_temperature_c=ambient_c,
            available_racks=available_racks,
            available_pcs=available_pcs,
            operating_mode=mode,
            alarm_count=alarms,
        )
        return site_row, pcs_output, rack_output

    def _ground_truth_at(self, timestamp: datetime) -> list[FaultGroundTruthRecord]:
        return [
            FaultGroundTruthRecord(
                timestamp_utc=timestamp,
                fault_id=event.fault_id,
                fault_type=event.fault_type,
                component_id=event.component_id,
                severity=event.severity,
                ground_truth_label=event.ground_truth_label,
            )
            for event in self.fault_engine.active_events(timestamp)
        ]

    def _ambient_series(self, value: float | Iterable[float] | None, count: int) -> list[float]:
        if value is None:
            return [self.config.thermal.ambient_temperature_c] * count
        if isinstance(value, int | float):
            return [float(value)] * count
        values = list(value)
        if len(values) != count:
            raise ValueError("ambient series length must match power request series")
        return values

    @staticmethod
    def _require_utc(value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("start_timestamp must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("start_timestamp must be UTC")
        return value.astimezone(UTC)
