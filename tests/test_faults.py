from datetime import timedelta

import pytest

from besspulse import BatterySimulator
from besspulse.ground_truth import FaultEvent, FaultType


def event(start, fault_type, component, *, duration_minutes=10, severity=1.0):
    return FaultEvent(
        fault_id=f"F-{fault_type}",
        fault_type=fault_type,
        component_id=component,
        start_timestamp=start + timedelta(minutes=5),
        end_timestamp=start + timedelta(minutes=5 + duration_minutes),
        severity=severity,
        progression_rate=0.0,
        ground_truth_label="injected_fault",
    )


def test_scheduled_fault_starts_and_ends_on_half_open_boundaries(start) -> None:
    fault = event(start, FaultType.RACK_OFFLINE, "PCS-01-RACK-01", duration_minutes=5)
    run = BatterySimulator(faults=[fault]).simulate(start, [4.0, 4.0, 4.0])
    labels_by_time = [row.timestamp_utc for row in run.fault_ground_truth]
    assert start not in labels_by_time
    assert start + timedelta(minutes=5) in labels_by_time
    assert start + timedelta(minutes=10) not in labels_by_time


def test_pcs_derating_reduces_delivered_and_available_power(start) -> None:
    fault = event(start, FaultType.PCS_DERATING, "PCS-01", severity=0.5)
    run = BatterySimulator(faults=[fault]).simulate(start, [20.0, 20.0])
    assert run.site_telemetry[1].available_power_mw < run.site_telemetry[0].available_power_mw
    assert run.site_telemetry[1].site_actual_power_mw < run.site_telemetry[0].site_actual_power_mw
    affected = [
        row
        for row in run.pcs_telemetry
        if row.timestamp_utc == start + timedelta(minutes=5) and row.pcs_id == "PCS-01"
    ]
    assert affected[0].actual_power_mw < 5.0


def test_rack_offline_removes_exact_rack_from_availability(start) -> None:
    fault = event(start, FaultType.RACK_OFFLINE, "PCS-01-RACK-01")
    run = BatterySimulator(faults=[fault]).simulate(start, [10.0, 10.0])
    assert run.site_telemetry[0].available_racks == 32
    assert run.site_telemetry[1].available_racks == 31
    row = next(
        item
        for item in run.rack_telemetry
        if item.timestamp_utc == start + timedelta(minutes=5) and item.rack_id == "PCS-01-RACK-01"
    )
    assert row.availability is False
    assert row.actual_power_mw == 0.0


def test_soc_sensor_bias_changes_observation_not_physical_state(start) -> None:
    target = "PCS-01-RACK-01"
    fault = event(start, FaultType.SOC_SENSOR_BIAS, target, severity=1.0)
    biased = BatterySimulator(faults=[fault])
    baseline = BatterySimulator()
    biased_run = biased.simulate(start, [0.0, 0.0])
    baseline_run = baseline.simulate(start, [0.0, 0.0])
    biased_row = next(
        row
        for row in biased_run.rack_telemetry
        if row.rack_id == target and row.timestamp_utc == start + timedelta(minutes=5)
    )
    baseline_row = next(
        row
        for row in baseline_run.rack_telemetry
        if row.rack_id == target and row.timestamp_utc == start + timedelta(minutes=5)
    )
    assert biased_row.soc == pytest.approx(baseline_row.soc + 0.15)
    assert biased.racks[0].state.true_soc == baseline.racks[0].state.true_soc


def test_thermal_drift_changes_observed_temperature(start) -> None:
    target = "PCS-01-RACK-01"
    fault = event(start, FaultType.THERMAL_DRIFT, target, severity=0.8)
    run = BatterySimulator(faults=[fault]).simulate(start, [0.0, 0.0])
    before, during = [row for row in run.rack_telemetry if row.rack_id == target]
    assert during.temperature_mean_c > before.temperature_mean_c + 10.0


def test_ground_truth_fields_are_not_telemetry_features(start) -> None:
    fault = event(start, FaultType.VOLTAGE_IMBALANCE, "PCS-01-RACK-01")
    run = BatterySimulator(faults=[fault]).simulate(start, [0.0, 0.0])
    forbidden = {"fault_id", "fault_type", "ground_truth_label", "future_fault_state"}
    for telemetry in (*run.site_telemetry, *run.pcs_telemetry, *run.rack_telemetry):
        assert forbidden.isdisjoint(telemetry.to_feature_dict())
    assert run.fault_ground_truth[0].fault_type is FaultType.VOLTAGE_IMBALANCE
