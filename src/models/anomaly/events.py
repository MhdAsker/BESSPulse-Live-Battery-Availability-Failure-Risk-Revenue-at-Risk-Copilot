"""Causal interval-to-event construction and deterministic fault matching."""

import hashlib
from typing import Any

import pandas as pd

from models.anomaly.common import json_safe_signals
from models.anomaly.config import AnomalyConfig


def create_anomaly_events(
    intervals: pd.DataFrame, config: AnomalyConfig | None = None
) -> pd.DataFrame:
    cfg = config or AnomalyConfig()
    required = [
        "timestamp_utc",
        "component_id",
        "component_type",
        "detector_name",
        "detector_version",
        "anomaly_score",
        "anomaly_flag",
        "supporting_signals",
    ]
    missing = [column for column in required if column not in intervals]
    if missing:
        raise ValueError(f"Missing anomaly event columns: {missing}")
    flagged = intervals.loc[intervals["anomaly_flag"]].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "anomaly_event_id",
                "component_id",
                "component_type",
                "detector_name",
                "detector_version",
                "start_timestamp",
                "end_timestamp",
                "duration_minutes",
                "peak_score",
                "mean_score",
                "trigger_count",
                "supporting_signals",
                "data_provenance",
            ]
        )
    flagged["timestamp_utc"] = pd.to_datetime(flagged["timestamp_utc"], utc=True)
    merge_gap = pd.Timedelta(cfg.merge_gap)
    rows: list[dict[str, Any]] = []
    group_keys = ["component_id", "component_type", "detector_name", "detector_version"]
    for identity, group in flagged.groupby(group_keys, sort=False):
        ordered = group.sort_values("timestamp_utc", kind="mergesort")
        event_group = ordered["timestamp_utc"].diff().gt(merge_gap).cumsum()
        for _, event in ordered.groupby(event_group, sort=False):
            start = event["timestamp_utc"].min()
            end = event["timestamp_utc"].max()
            raw_id = "|".join([*map(str, identity), start.isoformat()])
            signals = sorted(
                {
                    signal
                    for value in event["supporting_signals"]
                    for signal in json_safe_signals(value)
                }
            )
            rows.append(
                {
                    "anomaly_event_id": "ANOM-" + hashlib.sha256(raw_id.encode()).hexdigest()[:16],
                    "component_id": identity[0],
                    "component_type": identity[1],
                    "detector_name": identity[2],
                    "detector_version": identity[3],
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "duration_minutes": (end - start).total_seconds() / 60,
                    "peak_score": float(event["anomaly_score"].max()),
                    "mean_score": float(event["anomaly_score"].mean()),
                    "trigger_count": len(event),
                    "supporting_signals": signals,
                    "data_provenance": "DERIVED",
                }
            )
    return (
        pd.DataFrame(rows).sort_values("start_timestamp", kind="mergesort").reset_index(drop=True)
    )


def _component_matches(anomaly_component: str, fault_component: str) -> bool:
    return (
        fault_component == "SITE"
        or anomaly_component == fault_component
        or (fault_component.startswith("PCS-") and anomaly_component.startswith(fault_component))
    )


def match_fault_events(
    anomaly_events: pd.DataFrame,
    fault_events: pd.DataFrame,
    config: AnomalyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or AnomalyConfig()
    early = pd.Timedelta(cfg.early_warning_window)
    grace = pd.Timedelta(cfg.post_fault_grace)
    anomalies = anomaly_events.sort_values("start_timestamp", kind="mergesort").copy()
    used: set[str] = set()
    attributable: set[str] = set()
    matches: list[dict[str, Any]] = []
    raw_faults = fault_events.sort_values("start_timestamp", kind="mergesort").to_dict(
        orient="records"
    )
    fault_records: list[dict[str, Any]] = [
        {str(key): value for key, value in record.items()} for record in raw_faults
    ]
    for fault in fault_records:
        fault_component = str(fault["component_id"])
        compatible = pd.Series(
            [
                _component_matches(str(component), fault_component)
                for component in anomalies["component_id"]
            ],
            index=anomalies.index,
        )
        eligible = anomalies.loc[
            compatible
            & anomalies["start_timestamp"].ge(fault["start_timestamp"] - early)
            & anomalies["start_timestamp"].le(fault["end_timestamp"] + grace)
            & ~anomalies["anomaly_event_id"].isin(used)
        ]
        all_attributable = anomalies.loc[
            compatible
            & anomalies["start_timestamp"].ge(fault["start_timestamp"] - early)
            & anomalies["start_timestamp"].le(fault["end_timestamp"] + grace)
        ]
        attributable.update(all_attributable["anomaly_event_id"].astype(str))
        if eligible.empty:
            matches.append(
                {
                    **fault,
                    "detected": False,
                    "anomaly_event_id": None,
                    "detection_class": "MISSED_FAULT",
                    "detection_delay_minutes": float("nan"),
                }
            )
            continue
        anomaly = eligible.iloc[0]
        used.add(str(anomaly["anomaly_event_id"]))
        delay = (anomaly["start_timestamp"] - fault["start_timestamp"]).total_seconds() / 60
        classification = (
            "EARLY_DETECTION"
            if delay < 0
            else "ONSET_DETECTION"
            if delay == 0
            else "LATE_DETECTION"
        )
        matches.append(
            {
                **fault,
                "detected": True,
                "anomaly_event_id": anomaly["anomaly_event_id"],
                "detection_class": classification,
                "detection_delay_minutes": delay,
            }
        )
    false_alerts = anomalies.loc[~anomalies["anomaly_event_id"].isin(attributable)].copy()
    false_alerts["match_status"] = "FALSE_ALERT"
    return pd.DataFrame(matches), false_alerts
