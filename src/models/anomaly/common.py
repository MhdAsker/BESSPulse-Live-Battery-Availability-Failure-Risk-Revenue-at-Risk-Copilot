"""Shared anomaly interval output and causal persistence helpers."""

from typing import Any

import numpy as np
import pandas as pd


def apply_consecutive_persistence(
    scores: pd.DataFrame,
    *,
    threshold: float,
    intervals: int,
) -> pd.DataFrame:
    result = scores.sort_values(["component_id", "timestamp_utc"], kind="mergesort").copy()
    raw = result["anomaly_score"].ge(threshold) & result["anomaly_score"].notna()
    run_groups = (~raw).groupby(result["component_id"]).cumsum()
    result["persistence_count"] = (
        raw.groupby([result["component_id"], run_groups], sort=False).cumsum().astype(int)
    )
    result["anomaly_flag"] = result["persistence_count"].ge(intervals)
    result["threshold"] = threshold
    return result


def interval_output(
    keys: pd.DataFrame,
    score: pd.Series,
    *,
    detector_name: str,
    detector_version: str,
    component_type: str = "RACK",
    supporting_signals: list[list[str]] | None = None,
    provenance: str = "DERIVED",
) -> pd.DataFrame:
    result = keys.rename(columns={"rack_id": "component_id"})[
        ["timestamp_utc", "component_id"]
    ].copy()
    result["component_type"] = component_type
    result["detector_name"] = detector_name
    result["detector_version"] = detector_version
    result["anomaly_score"] = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    result["supporting_signals"] = supporting_signals or [[] for _ in range(len(result))]
    result["feature_set_version"] = "v1"
    result["data_provenance"] = provenance
    return result


def finite_or_nan(value: float) -> float:
    return value if np.isfinite(value) else float("nan")


def json_safe_signals(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return []
