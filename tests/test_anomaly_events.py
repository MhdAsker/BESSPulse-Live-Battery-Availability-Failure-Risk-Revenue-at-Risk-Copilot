import pandas as pd
import pytest

from models.anomaly.common import apply_consecutive_persistence
from models.anomaly.config import AnomalyConfig
from models.anomaly.evaluate import healthy_duration_days
from models.anomaly.events import create_anomaly_events, match_fault_events


def _intervals() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "component_id": ["RACK-01"] * 6,
            "component_type": ["RACK"] * 6,
            "detector_name": ["test"] * 6,
            "detector_version": ["v1"] * 6,
            "anomaly_score": [0.0, 2.0, 2.0, 0.0, 2.0, 2.0],
            "supporting_signals": [["temperature"]] * 6,
        }
    )


def test_persistence_is_causal_and_event_gap_is_merged() -> None:
    persisted = apply_consecutive_persistence(_intervals(), threshold=1.0, intervals=2)
    assert persisted["anomaly_flag"].tolist() == [False, False, True, False, False, True]
    events = create_anomaly_events(persisted, AnomalyConfig(merge_gap="15min"))
    assert len(events) == 1
    assert events.iloc[0]["trigger_count"] == 2


def test_event_matching_classifies_detection_and_false_alerts() -> None:
    persisted = apply_consecutive_persistence(_intervals(), threshold=1.0, intervals=2)
    events = create_anomaly_events(persisted, AnomalyConfig(merge_gap="0min"))
    faults = pd.DataFrame(
        {
            "fault_id": ["F1"],
            "component_id": ["RACK-01"],
            "start_timestamp": [pd.Timestamp("2026-01-01 00:15", tz="UTC")],
            "end_timestamp": [pd.Timestamp("2026-01-01 00:20", tz="UTC")],
        }
    )
    matches, false_alerts = match_fault_events(
        events, faults, AnomalyConfig(early_warning_window="10min", post_fault_grace="0min")
    )
    assert matches.iloc[0]["detection_class"] == "EARLY_DETECTION"
    assert matches.iloc[0]["detection_delay_minutes"] == -5
    assert len(false_alerts) == 1


def test_healthy_duration_uses_time_not_row_count() -> None:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(days=2)
    faults = pd.DataFrame(
        {
            "component_id": ["SITE"],
            "start_timestamp": [start + pd.Timedelta(hours=6)],
            "end_timestamp": [start + pd.Timedelta(hours=18)],
        }
    )
    assert healthy_duration_days(start, end, faults) == 1.5


def test_long_gap_stays_separate_and_event_statistics_are_exact() -> None:
    intervals = _intervals()
    intervals["anomaly_flag"] = [False, True, True, False, True, True]
    events = create_anomaly_events(intervals, AnomalyConfig(merge_gap="5min"))
    assert len(events) == 2
    assert events["duration_minutes"].tolist() == [5.0, 5.0]
    assert events["peak_score"].tolist() == [2.0, 2.0]
    assert events["mean_score"].tolist() == [2.0, 2.0]


def test_onset_late_and_missed_matching_have_correct_delays() -> None:
    start = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    faults = pd.DataFrame(
        {
            "fault_id": ["ONSET", "LATE", "MISSED"],
            "component_id": ["RACK-01", "RACK-02", "RACK-03"],
            "start_timestamp": [start, start, start],
            "end_timestamp": [start + pd.Timedelta("1h")] * 3,
        }
    )
    events = pd.DataFrame(
        {
            "anomaly_event_id": ["A", "B"],
            "component_id": ["RACK-01", "RACK-02"],
            "start_timestamp": [start, start + pd.Timedelta("15min")],
            "end_timestamp": [start, start + pd.Timedelta("15min")],
        }
    )
    matches, false_alerts = match_fault_events(events, faults)
    assert matches["detection_class"].tolist() == [
        "ONSET_DETECTION",
        "LATE_DETECTION",
        "MISSED_FAULT",
    ]
    assert matches.loc[0, "detection_delay_minutes"] == 0
    assert matches.loc[1, "detection_delay_minutes"] == 15
    assert pd.isna(matches.loc[2, "detection_delay_minutes"])
    assert false_alerts.empty


def test_zero_healthy_duration_is_explicit_error() -> None:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(days=1)
    faults = pd.DataFrame(
        {
            "component_id": ["SITE"],
            "start_timestamp": [start],
            "end_timestamp": [end],
        }
    )
    with pytest.raises(ValueError, match="zero healthy duration"):
        healthy_duration_days(start, end, faults)
