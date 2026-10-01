from datetime import timedelta

import pytest

from besspulse import BatterySimulator, SimulationConfig


def test_hierarchy_has_configured_component_counts() -> None:
    simulator = BatterySimulator()
    assert len(simulator.pcs_units) == 4
    assert len(simulator.racks) == 32
    assert all(len(pcs.racks) == 8 for pcs in simulator.pcs_units)


def test_soc_never_exceeds_bounds_under_sustained_requests(start) -> None:
    discharge = BatterySimulator()
    discharge.simulate(start, [100.0] * 96)
    assert all(0.10 <= rack.state.true_soc <= 0.90 for rack in discharge.racks)
    charge = BatterySimulator()
    charge.simulate(start, [-100.0] * 96)
    assert all(0.10 <= rack.state.true_soc <= 0.90 for rack in charge.racks)


def test_discharge_energy_and_soc_accounting_are_consistent(start) -> None:
    simulator = BatterySimulator()
    initial_soc = simulator.racks[0].state.true_soc
    run = simulator.simulate(start, [8.0])
    rack_row = run.rack_telemetry[0]
    rack = simulator.racks[0]
    delivered = rack_row.actual_power_mw * (5 / 60)
    efficiency = rack_row.rte**0.5
    expected_soc = initial_soc - delivered / (efficiency * rack.nominal_capacity_mwh)
    assert rack.state.true_soc == pytest.approx(expected_soc, rel=1e-6)
    assert rack_row.discharge_energy_mwh == pytest.approx(delivered)
    assert rack_row.cumulative_throughput_mwh == pytest.approx(delivered)


def test_site_and_component_power_limits_are_respected(start) -> None:
    simulator = BatterySimulator()
    run = simulator.simulate(start, [100.0, -100.0])
    assert all(abs(row.site_actual_power_mw) <= 20.0 + 1e-9 for row in run.site_telemetry)
    assert all(abs(row.actual_power_mw) <= 0.625 + 1e-9 for row in run.rack_telemetry)
    assert all(abs(row.actual_power_mw) <= 5.0 + 1e-9 for row in run.pcs_telemetry)


def test_timestamps_are_utc_and_follow_interval(start) -> None:
    run = BatterySimulator().simulate(start, [0.0, 0.0, 0.0])
    timestamps = [row.timestamp_utc for row in run.site_telemetry]
    assert all(value.utcoffset() == timedelta(0) for value in timestamps)
    assert timestamps[1] - timestamps[0] == timedelta(minutes=5)
    assert timestamps[2] - timestamps[1] == timedelta(minutes=5)


def test_identical_seed_and_configuration_are_reproducible(start) -> None:
    config = SimulationConfig(simulation_seed=7)
    first = BatterySimulator(config).simulate(start, [3.0, -2.0, 7.0])
    second = BatterySimulator(config).simulate(start, [3.0, -2.0, 7.0])
    assert first == second


def test_temperature_increases_with_sustained_power(start) -> None:
    run = BatterySimulator().simulate(start, [20.0] * 12)
    first = run.rack_telemetry[0].temperature_mean_c
    last = run.rack_telemetry[-32].temperature_mean_c
    assert last > first
