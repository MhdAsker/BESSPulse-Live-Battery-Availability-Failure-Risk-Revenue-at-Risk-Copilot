"""Interval/event metrics and validation-only threshold selection."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from models.anomaly.common import apply_consecutive_persistence
from models.anomaly.config import AnomalyConfig
from models.anomaly.events import create_anomaly_events, match_fault_events


def interval_truth(keys: pd.DataFrame, faults: pd.DataFrame) -> pd.Series:
    truth = pd.Series(False, index=keys.index)
    for fault in faults.to_dict(orient="records"):
        component = str(fault["component_id"])
        affected = (
            pd.Series(True, index=keys.index)
            if component == "SITE"
            else keys["rack_id"].astype(str).eq(component)
            if "RACK" in component
            else keys["rack_id"].astype(str).str.startswith(component)
        )
        active = keys["timestamp_utc"].between(
            fault["start_timestamp"], fault["end_timestamp"], inclusive="left"
        )
        truth |= affected & active
    return truth


def healthy_duration_days(
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    faults: pd.DataFrame,
) -> float:
    total = (evaluation_end - evaluation_start).total_seconds()
    clipped: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fault in faults.to_dict(orient="records"):
        start = max(evaluation_start, fault["start_timestamp"])
        end = min(evaluation_end, fault["end_timestamp"])
        if start < end:
            clipped.append((start, end))
    clipped.sort()
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    fault_seconds = sum((end - start).total_seconds() for start, end in merged)
    healthy = total - fault_seconds
    if healthy <= 0:
        raise ValueError("False-alert rate is undefined with zero healthy duration")
    return healthy / 86400


def detector_metrics(
    intervals: pd.DataFrame,
    keys: pd.DataFrame,
    fault_events: pd.DataFrame,
    config: AnomalyConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = config or AnomalyConfig()
    aligned_keys = intervals[["timestamp_utc", "component_id"]].rename(
        columns={"component_id": "rack_id"}
    )
    truth = interval_truth(aligned_keys, fault_events).to_numpy(dtype=bool)
    flag = intervals["anomaly_flag"].to_numpy(dtype=bool)
    score = intervals["anomaly_score"].fillna(0).to_numpy(dtype=float)
    events = create_anomaly_events(intervals, cfg)
    matches, false_alerts = match_fault_events(events, fault_events, cfg)
    delays = matches.loc[matches["detected"], "detection_delay_minutes"].astype(float)
    healthy_days = healthy_duration_days(
        keys["timestamp_utc"].min(), keys["timestamp_utc"].max(), fault_events
    )
    metrics: dict[str, Any] = {
        "interval_precision": float(precision_score(truth, flag, zero_division=0)),
        "interval_recall": float(recall_score(truth, flag, zero_division=0)),
        "interval_f1": float(f1_score(truth, flag, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(truth, score))
        if truth.any()
        else float("nan"),
        "fault_event_count": len(matches),
        "detected_events": int(matches["detected"].sum()),
        "missed_events": int((~matches["detected"]).sum()),
        "event_detection_rate": float(matches["detected"].mean()) if len(matches) else float("nan"),
        "false_event_count": len(false_alerts),
        "healthy_evaluation_days": healthy_days,
        "false_alerts_per_day": len(false_alerts) / healthy_days,
        "mean_detection_delay_minutes": float(delays.mean()) if len(delays) else float("nan"),
        "median_detection_delay_minutes": float(delays.median()) if len(delays) else float("nan"),
        "p90_detection_delay_minutes": float(delays.quantile(0.9))
        if len(delays) >= 3
        else float("nan"),
        "minimum_detection_delay_minutes": float(delays.min()) if len(delays) else float("nan"),
        "maximum_detection_delay_minutes": float(delays.max()) if len(delays) else float("nan"),
    }
    return metrics, matches, false_alerts


def select_threshold_persistence(
    scores: pd.DataFrame,
    keys: pd.DataFrame,
    faults: pd.DataFrame,
    thresholds: tuple[float, ...],
    config: AnomalyConfig,
) -> tuple[float, int, list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    for threshold in thresholds:
        for persistence in config.persistence_candidates:
            intervals = apply_consecutive_persistence(
                scores, threshold=threshold, intervals=persistence
            )
            metrics, _, _ = detector_metrics(intervals, keys, faults, config)
            comparisons.append(
                {
                    "threshold": threshold,
                    "persistence": persistence,
                    **metrics,
                }
            )
    within_budget = [
        row for row in comparisons if row["false_alerts_per_day"] <= config.max_false_alerts_per_day
    ]
    pool = within_budget or comparisons
    selected = min(
        pool,
        key=lambda row: (
            -float(row["event_detection_rate"]),
            float(row["median_detection_delay_minutes"])
            if np.isfinite(row["median_detection_delay_minutes"])
            else float("inf"),
            int(row["persistence"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), int(selected["persistence"]), comparisons
