"""PCS hierarchy object and telemetry aggregation."""

from dataclasses import dataclass
from datetime import datetime

from besspulse.schemas import PCSTelemetry, RackTelemetry
from besspulse.simulator.rack import Rack


@dataclass
class PCS:
    pcs_id: str
    racks: list[Rack]

    def telemetry(
        self,
        timestamp: datetime,
        requested_power_mw: float,
        rack_rows: list[RackTelemetry],
        derating_factor: float,
    ) -> PCSTelemetry:
        actual = sum(row.actual_power_mw for row in rack_rows)
        available = any(row.availability for row in rack_rows)
        total_abs_power = sum(abs(row.actual_power_mw) for row in rack_rows)
        efficiency = (
            sum(row.rte**0.5 * abs(row.actual_power_mw) for row in rack_rows) / total_abs_power
            if total_abs_power > 0
            else 0.96
        )
        temperature = (
            sum(row.temperature_mean_c for row in rack_rows) / len(rack_rows) if rack_rows else 0.0
        )
        alarm = any(row.alarm_code is not None for row in rack_rows)
        return PCSTelemetry(
            timestamp_utc=timestamp,
            pcs_id=self.pcs_id,
            requested_power_mw=requested_power_mw,
            actual_power_mw=actual,
            efficiency=efficiency,
            temperature_c=temperature,
            availability=available,
            derating_factor=derating_factor,
            alarm_state=alarm,
        )
