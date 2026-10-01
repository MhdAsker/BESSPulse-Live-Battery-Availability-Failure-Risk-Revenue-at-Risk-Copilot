from datetime import UTC, datetime

import pandas as pd
import pytest

from availability.engine import calculate_availability_bundle
from availability.metrics import (
    availability_summary,
    component_downtime,
    create_availability_events,
    interval_weights_hours,
    time_weighted_mean,
)
from besspulse import BatterySimulator


def history_frames(count: int = 4):
    run = BatterySimulator().simulate(
        datetime(2026, 1, 1, tzinfo=UTC), [2.0, -2.0, 0.0, 2.0][:count]
    )
    return (
        pd.DataFrame(row.model_dump() for row in run.site_telemetry),
        pd.DataFrame(row.model_dump() for row in run.pcs_telemetry),
        pd.DataFrame(row.model_dump() for row in run.rack_telemetry),
    )


def test_availability_history_is_chronological_and_deterministic() -> None:
    site, pcs, racks = history_frames()
    shuffled = site.sample(frac=1, random_state=42)
    first = calculate_availability_bundle(shuffled, pcs, racks)
    second = calculate_availability_bundle(shuffled, pcs, racks)
    assert first.snapshots["timestamp_utc"].is_monotonic_increasing
    pd.testing.assert_frame_equal(first.snapshots, second.snapshots)


def test_missing_current_row_does_not_use_future_state() -> None:
    site, pcs, racks = history_frames()
    target_time = site.loc[1, "timestamp_utc"]
    rack_id = racks.loc[0, "rack_id"]
    racks = racks.loc[~(racks["timestamp_utc"].eq(target_time) & racks["rack_id"].eq(rack_id))]
    bundle = calculate_availability_bundle(site, pcs, racks)
    row = bundle.snapshots.loc[bundle.snapshots["timestamp_utc"].eq(target_time)].iloc[0]
    assert row["unknown_racks"] == 1


def test_irregular_time_weighting_is_elapsed_time_based() -> None:
    timestamps = pd.Series(
        pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:05Z", "2026-01-01T00:20Z"], utc=True)
    )
    values = pd.Series([1.0, 0.0, 0.0])
    weights = interval_weights_hours(timestamps)
    assert weights.iloc[:2].tolist() == pytest.approx([5 / 60, 15 / 60])
    assert time_weighted_mean(values, timestamps) == pytest.approx(1 / 6)


def test_downtime_duration_events_and_longest_outage() -> None:
    times = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    capabilities = pd.DataFrame(
        {
            "timestamp_utc": times,
            "component_id": ["R1"] * 5,
            "state": ["AVAILABLE", "UNAVAILABLE", "UNAVAILABLE", "AVAILABLE", "UNAVAILABLE"],
            "nominal_power_mw": [1.0] * 5,
            "available_discharge_power_mw": [1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    downtime = component_downtime(capabilities).iloc[0]
    assert downtime["downtime_minutes"] == 15
    assert downtime["number_of_outage_events"] == 2
    assert downtime["longest_outage_minutes"] == 10
    events = create_availability_events(capabilities)
    assert events["event_type"].tolist() == ["DOWNTIME", "DOWNTIME"]
    assert events["duration_minutes"].tolist() == [10.0, 5.0]


def test_derating_event_is_distinct_from_downtime() -> None:
    capabilities = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "component_id": ["R1"] * 3,
            "state": ["AVAILABLE"] * 3,
            "nominal_power_mw": [1.0] * 3,
            "available_discharge_power_mw": [1.0, 0.5, 0.5],
        }
    )
    events = create_availability_events(capabilities)
    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "DERATED"
    assert events.iloc[0]["duration_minutes"] == 10


def test_summary_separates_request_support_from_delivery_success() -> None:
    site, pcs, racks = history_frames()
    site.loc[0, "site_actual_power_mw"] = 0.0
    snapshots = calculate_availability_bundle(site, pcs, racks).snapshots
    summary = availability_summary(snapshots)
    assert summary["request_support_rate"] == 1.0
    assert summary["observed_delivery_success_rate"] < 1.0
    assert summary["active_requested_power_intervals"] == 3
