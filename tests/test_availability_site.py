from datetime import UTC, datetime

import pandas as pd
import pytest

from availability.engine import calculate_availability_snapshot
from besspulse import BatterySimulator


def telemetry(request: float = 2.0):
    run = BatterySimulator().simulate(datetime(2026, 1, 1, tzinfo=UTC), [request])
    site = run.site_telemetry[0].model_dump()
    pcs = pd.DataFrame(row.model_dump() for row in run.pcs_telemetry)
    racks = pd.DataFrame(row.model_dump() for row in run.rack_telemetry)
    return site, pcs, racks


def test_site_baseline_conserves_power_energy_and_counts() -> None:
    site, pcs, racks = telemetry()
    snapshot, rack_caps, pcs_caps = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.technical_availability == 1
    assert snapshot.available_discharge_power_mw <= snapshot.rated_power_mw
    assert snapshot.available_discharge_power_mw == pytest.approx(
        sum(item.available_discharge_power_mw for item in pcs_caps)
    )
    assert snapshot.available_discharge_energy_mwh == pytest.approx(
        sum(item.available_discharge_energy_mwh for item in rack_caps)
    )
    assert snapshot.available_racks == 32
    assert snapshot.available_pcs == 4


def test_one_offline_rack_reduces_technical_power_and_energy() -> None:
    site, pcs, racks = telemetry()
    racks.loc[0, "availability"] = False
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.technical_availability == pytest.approx(31 / 32)
    assert snapshot.available_discharge_power_mw < snapshot.rated_power_mw
    assert snapshot.available_discharge_energy_mwh < 15.36


def test_derating_can_lower_power_while_technical_availability_is_one() -> None:
    site, pcs, racks = telemetry()
    pcs.loc[pcs["pcs_id"].eq("PCS-01"), "derating_factor"] = 0.5
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.technical_availability == 1
    assert snapshot.discharge_power_availability < 1
    assert "POWER_LIMIT" in snapshot.limiting_factors


def test_low_soc_can_constrain_energy_while_technical_availability_is_one() -> None:
    site, pcs, racks = telemetry()
    racks["soc"] = 0.20
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.technical_availability == 1
    assert snapshot.discharge_power_availability == 1
    assert snapshot.discharge_energy_availability < 0.13


def test_high_soc_separates_charge_and_discharge_capability() -> None:
    site, pcs, racks = telemetry(-2.0)
    racks["soc"] = 0.899
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.discharge_energy_availability > 0.99
    assert snapshot.charge_energy_availability < 0.01
    assert snapshot.available_discharge_power_mw == 20
    assert snapshot.available_charge_power_mw < 1


def test_capability_can_be_one_while_observed_delivery_fails() -> None:
    site, pcs, racks = telemetry(10.0)
    site["site_actual_power_mw"] = 9.0
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.requested_power_availability == 1
    assert snapshot.observed_delivery_success is False
    assert snapshot.delivery_ratio == 0.9
    assert snapshot.capability_delivery_gap == pytest.approx(-0.1)


def test_missing_component_is_unknown_not_available() -> None:
    site, pcs, racks = telemetry()
    racks = racks.iloc[1:].copy()
    snapshot, _, _ = calculate_availability_snapshot(site, pcs, racks)
    assert snapshot.unknown_racks == 1
    assert snapshot.known_component_fraction < 1
    assert snapshot.technical_availability == pytest.approx(31 / 32)
    assert snapshot.data_provenance == "DERIVED ENGINEERING ANALYTIC"
