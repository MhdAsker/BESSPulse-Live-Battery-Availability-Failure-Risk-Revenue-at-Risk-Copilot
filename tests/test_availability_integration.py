from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from availability.engine import calculate_availability_bundle
from availability.metrics import availability_summary, component_downtime
from availability.storage import persist_availability_snapshots
from besspulse import BatterySimulator
from besspulse.database import Base, create_engine
from besspulse.ground_truth import FaultEvent, FaultType


def test_deterministic_availability_end_to_end() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    faults = [
        FaultEvent(
            fault_id="RACK-OUT",
            fault_type=FaultType.RACK_OFFLINE,
            component_id="PCS-01-RACK-01",
            start_timestamp=start + timedelta(minutes=20),
            end_timestamp=start + timedelta(minutes=40),
            severity=1.0,
            progression_rate=0.0,
            ground_truth_label="RACK_OFFLINE",
        ),
        FaultEvent(
            fault_id="PCS-DERATE",
            fault_type=FaultType.PCS_DERATING,
            component_id="PCS-02",
            start_timestamp=start + timedelta(minutes=45),
            end_timestamp=start + timedelta(minutes=65),
            severity=0.5,
            progression_rate=0.0,
            ground_truth_label="PCS_DERATING",
        ),
    ]
    run = BatterySimulator(faults=faults).simulate(start, [8.0, -8.0] * 10)
    site = pd.DataFrame(row.model_dump() for row in run.site_telemetry)
    pcs = pd.DataFrame(row.model_dump() for row in run.pcs_telemetry)
    racks = pd.DataFrame(row.model_dump() for row in run.rack_telemetry)
    low_time = site.loc[14, "timestamp_utc"]
    high_time = site.loc[15, "timestamp_utc"]
    racks.loc[racks["timestamp_utc"].eq(low_time), "soc"] = 0.101
    racks.loc[racks["timestamp_utc"].eq(high_time), "soc"] = 0.899
    first = calculate_availability_bundle(site, pcs, racks)
    second = calculate_availability_bundle(site, pcs, racks)
    pd.testing.assert_frame_equal(first.snapshots, second.snapshots)
    baseline = first.snapshots.iloc[0]
    outage = first.snapshots.loc[
        first.snapshots["timestamp_utc"].eq(start + timedelta(minutes=20))
    ].iloc[0]
    derated = first.snapshots.loc[
        first.snapshots["timestamp_utc"].eq(start + timedelta(minutes=45))
    ].iloc[0]
    low = first.snapshots.loc[first.snapshots["timestamp_utc"].eq(low_time)].iloc[0]
    high = first.snapshots.loc[first.snapshots["timestamp_utc"].eq(high_time)].iloc[0]
    assert baseline["technical_availability"] == 1
    assert outage["technical_availability"] < baseline["technical_availability"]
    assert outage["available_discharge_energy_mwh"] < baseline["available_discharge_energy_mwh"]
    assert derated["power_availability"] < 1
    assert derated["technical_availability"] == 1
    assert low["discharge_energy_availability"] < 0.01
    assert high["charge_energy_availability"] < 0.01
    assert first.snapshots["requested_power_availability"].notna().all()
    assert first.snapshots["available_discharge_power_mw"].le(20).all()
    assert first.pcs_capabilities["available_discharge_power_mw"].le(5).all()
    assert not any(
        column in first.snapshots
        for column in ["fault_type", "fault_id", "ground_truth_label", "severity"]
    )
    downtime = component_downtime(first.rack_capabilities)
    assert downtime["downtime_minutes"].sum() > 0
    summary = availability_summary(first.snapshots)
    assert summary["snapshot_count"] == 20
    database = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(database)
    with Session(database) as session:
        assert persist_availability_snapshots(session, first.snapshots) == 20
        assert persist_availability_snapshots(session, first.snapshots) == 0
