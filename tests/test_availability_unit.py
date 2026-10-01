from datetime import UTC, datetime, timedelta

import pytest

from availability.config import AvailabilityConfig
from availability.pcs import calculate_pcs_availability
from availability.rack import calculate_rack_availability, thermal_derating_factor
from availability.requested_power import (
    calculate_requested_power_availability,
    power_direction,
    sustainable_power,
)
from availability.schemas import ComponentState
from besspulse.config import BatteryConfig, SimulationConfig


def rack_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "rack_id": "PCS-01-RACK-01",
        "pcs_id": "PCS-01",
        "availability": True,
        "operating_state": "IDLE",
        "soc": 0.5,
        "soh_proxy": 1.0,
        "temperature_mean_c": 25.0,
    }
    row.update(changes)
    return row


def test_healthy_rack_has_separate_power_and_energy_capability() -> None:
    config = AvailabilityConfig()
    result = calculate_rack_availability(rack_row(), config)
    assert result.state == ComponentState.AVAILABLE
    assert result.available_discharge_power_mw == config.nominal_rack_power_mw
    assert result.available_charge_power_mw == config.nominal_rack_power_mw
    assert result.available_discharge_energy_mwh == pytest.approx(0.48)
    assert result.available_charge_energy_mwh == pytest.approx(0.5 / 0.96)


@pytest.mark.parametrize("availability,pcs_available", [(False, True), (True, False)])
def test_unavailable_rack_or_parent_contributes_zero(
    availability: bool, pcs_available: bool
) -> None:
    result = calculate_rack_availability(
        rack_row(availability=availability),
        AvailabilityConfig(),
        parent_pcs_available=pcs_available,
    )
    assert result.state == ComponentState.UNAVAILABLE
    assert result.available_discharge_power_mw == 0
    assert result.available_charge_energy_mwh == 0


def test_thermal_derating_lowers_power_without_technical_outage() -> None:
    config = AvailabilityConfig()
    temperature = (config.simulation.thermal.derating_start_c + 55.0) / 2
    result = calculate_rack_availability(rack_row(temperature_mean_c=temperature), config)
    assert result.technical_available is True
    assert result.thermal_derating_factor == pytest.approx(0.5)
    assert result.available_discharge_power_mw == pytest.approx(config.nominal_rack_power_mw / 2)


@pytest.mark.parametrize(
    "temperature,expected", [(30.0, 1.0), (42.0, 1.0), (48.5, 0.5), (55.0, 0.0)]
)
def test_thermal_derating_bounds(temperature: float, expected: float) -> None:
    assert thermal_derating_factor(temperature, AvailabilityConfig()) == pytest.approx(expected)


def test_low_soc_reduces_discharge_energy_but_not_charge_capability() -> None:
    config = AvailabilityConfig()
    result = calculate_rack_availability(rack_row(soc=config.minimum_soc), config)
    assert result.technical_available is True
    assert result.available_discharge_energy_mwh == 0
    assert result.available_discharge_power_mw == 0
    assert result.available_charge_power_mw == config.nominal_rack_power_mw
    assert "SOC_LOW" in result.limiting_factors


def test_high_soc_reduces_charge_headroom_but_preserves_discharge() -> None:
    config = AvailabilityConfig()
    result = calculate_rack_availability(rack_row(soc=config.maximum_soc), config)
    assert result.available_charge_energy_mwh == 0
    assert result.available_charge_power_mw == 0
    assert result.available_discharge_power_mw == config.nominal_rack_power_mw
    assert result.available_discharge_energy_mwh > 0


def test_capacity_fade_lowers_energy_not_technical_state() -> None:
    full = calculate_rack_availability(rack_row(), AvailabilityConfig())
    faded = calculate_rack_availability(rack_row(soh_proxy=0.8), AvailabilityConfig())
    assert faded.technical_available is True
    assert faded.available_discharge_energy_mwh == pytest.approx(
        full.available_discharge_energy_mwh * 0.8
    )
    assert "CAPACITY_FADE" in faded.limiting_factors


@pytest.mark.parametrize("missing", ["soc", "soh_proxy", "availability", "temperature_mean_c"])
def test_missing_required_rack_telemetry_is_unknown(missing: str) -> None:
    row = rack_row()
    row.pop(missing)
    result = calculate_rack_availability(row, AvailabilityConfig())
    assert result.state == ComponentState.UNKNOWN
    assert result.technical_available is None
    assert result.available_discharge_power_mw == 0


def test_stale_telemetry_is_unknown_and_future_telemetry_is_rejected() -> None:
    config = AvailabilityConfig(maximum_telemetry_staleness="5min")
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    stale = calculate_rack_availability(
        rack_row(timestamp_utc=timestamp),
        config,
        reference_timestamp=timestamp + timedelta(minutes=6),
    )
    assert stale.state == ComponentState.UNKNOWN
    with pytest.raises(ValueError, match="future"):
        calculate_rack_availability(
            rack_row(timestamp_utc=timestamp + timedelta(minutes=1)),
            config,
            reference_timestamp=timestamp,
        )


def test_pcs_aggregation_respects_converter_limit_and_rack_energy() -> None:
    config = AvailabilityConfig()
    racks = [
        calculate_rack_availability(rack_row(rack_id=f"PCS-01-RACK-{index:02d}"), config)
        for index in range(1, 9)
    ]
    result = calculate_pcs_availability(
        "PCS-01",
        datetime(2026, 1, 1, tzinfo=UTC),
        racks,
        {"availability": True, "derating_factor": 0.5, "requested_power_mw": 2.0},
        config,
    )
    assert result.technical_available is True
    assert result.available_discharge_power_mw == pytest.approx(2.5)
    assert result.available_discharge_energy_mwh == pytest.approx(
        sum(rack.available_discharge_energy_mwh for rack in racks)
    )
    assert result.available_discharge_power_mw <= sum(
        rack.available_discharge_power_mw for rack in racks
    )


def test_offline_and_unknown_pcs_block_dispatch() -> None:
    config = AvailabilityConfig()
    racks = [calculate_rack_availability(rack_row(), config)]
    offline = calculate_pcs_availability(
        "PCS-01",
        datetime(2026, 1, 1, tzinfo=UTC),
        racks,
        {"availability": False, "derating_factor": 1.0},
        config,
    )
    unknown = calculate_pcs_availability(
        "PCS-01", datetime(2026, 1, 1, tzinfo=UTC), racks, None, config
    )
    assert offline.state == ComponentState.UNAVAILABLE
    assert unknown.state == ComponentState.UNKNOWN
    assert offline.available_discharge_power_mw == unknown.available_discharge_power_mw == 0


@pytest.mark.parametrize(
    "requested,expected",
    [(2.0, "DISCHARGE"), (-2.0, "CHARGE"), (0.0, "IDLE"), (0.1, "IDLE")],
)
def test_power_direction_uses_simulator_sign_convention(requested: float, expected: str) -> None:
    assert power_direction(requested, 0.1) == expected


@pytest.mark.parametrize(
    "requested,discharge,charge,expected",
    [(10.0, 20.0, 5.0, 1.0), (20.0, 15.0, 20.0, 0.75), (-10.0, 20.0, 5.0, 0.5)],
)
def test_requested_power_selects_directional_capability(
    requested: float, discharge: float, charge: float, expected: float
) -> None:
    result = calculate_requested_power_availability(
        requested,
        available_discharge_power_mw=discharge,
        available_charge_power_mw=charge,
        available_discharge_energy_mwh=20.0,
        available_charge_energy_mwh=20.0,
        config=AvailabilityConfig(),
    )
    assert result.requested_power_availability == pytest.approx(expected)


def test_idle_request_is_not_applicable() -> None:
    result = calculate_requested_power_availability(
        0.0,
        available_discharge_power_mw=20,
        available_charge_power_mw=20,
        available_discharge_energy_mwh=20,
        available_charge_energy_mwh=20,
        config=AvailabilityConfig(),
    )
    assert result.requested_power_is_active is False
    assert result.requested_power_availability is None
    assert result.sustainable_request_duration_hours is None


def test_request_energy_sufficiency_and_sustainable_power() -> None:
    result = calculate_requested_power_availability(
        10.0,
        available_discharge_power_mw=20,
        available_charge_power_mw=20,
        available_discharge_energy_mwh=5,
        available_charge_energy_mwh=20,
        config=AvailabilityConfig(requested_energy_duration_hours=1.0),
    )
    assert result.requested_power_availability == 1
    assert result.requested_energy_availability == 0.5
    assert result.sustainable_request_duration_hours == 0.5
    assert sustainable_power(20, 5, 2) == 2.5


def test_configuration_is_not_hard_coded_to_default_asset() -> None:
    simulation = SimulationConfig(
        battery=BatteryConfig(
            site_rated_power_mw=12,
            site_rated_energy_mwh=18,
            pcs_count=3,
            racks_per_pcs=2,
            soc_min=0.2,
            soc_max=0.8,
            initial_soc=0.5,
        )
    )
    config = AvailabilityConfig(simulation=simulation)
    assert config.total_racks == 6
    assert config.nominal_rack_power_mw == 2
    assert config.nominal_rack_energy_mwh == 3
    low = calculate_rack_availability(rack_row(soc=0.2), config)
    assert low.available_discharge_energy_mwh == 0


def test_ground_truth_and_model_context_do_not_change_rack_capability() -> None:
    base = calculate_rack_availability(rack_row(), AvailabilityConfig())
    contextual = calculate_rack_availability(
        rack_row(
            fault_type="RACK_OFFLINE",
            ground_truth_label="fault",
            severity=1.0,
            failure_probability_6h=0.99,
            anomaly_score=100.0,
        ),
        AvailabilityConfig(),
    )
    assert contextual == base
