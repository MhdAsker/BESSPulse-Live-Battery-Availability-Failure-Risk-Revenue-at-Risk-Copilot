"""Central temporal, identity, numerical, and leakage guards."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

EXPLICIT_PROHIBITED_COLUMNS = frozenset(
    {
        "fault_id",
        "fault_type",
        "fault_severity",
        "ground_truth_label",
        "future_fault_state",
        "failure_within_6h",
        "failure_within_12h",
        "failure_within_24h",
        "target",
        "known_fault",
    }
)


def is_prohibited_column(column: str) -> bool:
    normalized = column.lower()
    return (
        normalized in EXPLICIT_PROHIBITED_COLUMNS
        or normalized.startswith("future_")
        or normalized.startswith("target_")
        or normalized.startswith("failure_within_")
        or normalized.startswith("ground_truth_")
    )


def validate_no_leakage(columns: Sequence[str]) -> None:
    """Reject explicit truth/target fields and documented defensive prefixes."""

    violations = {column for column in columns if is_prohibited_column(column)}
    if violations:
        raise ValueError(f"Leakage-prohibited feature columns: {sorted(violations)}")


def prepare_temporal_frame(
    frame: pd.DataFrame,
    entity_columns: Sequence[str],
    *,
    timestamp_column: str = "timestamp_utc",
) -> pd.DataFrame:
    """Validate aware timestamps and uniqueness, then sort deterministically."""

    required = [timestamp_column, *entity_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing key columns: {missing}")
    result = frame.copy()
    timestamps = result[timestamp_column]
    if timestamps.isna().any():
        raise ValueError("timestamp contains missing values")
    for value in timestamps:
        if not isinstance(value, pd.Timestamp):
            value = pd.Timestamp(value)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
    result[timestamp_column] = pd.to_datetime(timestamps, utc=True)
    keys = [timestamp_column, *entity_columns]
    duplicate = result.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate feature identity for keys {keys}")
    return result.sort_values([*entity_columns, timestamp_column], kind="mergesort").reset_index(
        drop=True
    )


def validate_no_infinite(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Feature matrix contains infinite values")


def validate_feature_frame(
    frame: pd.DataFrame,
    entity_columns: Sequence[str],
    *,
    timestamp_column: str = "timestamp_utc",
) -> None:
    validate_no_leakage(list(frame.columns))
    prepare_temporal_frame(frame, entity_columns, timestamp_column=timestamp_column)
    validate_no_infinite(frame)


def trailing_statistic(
    values: pd.Series,
    timestamps: pd.Series,
    window: str,
    statistic: Literal["mean", "std", "sum"],
    *,
    min_periods: int,
    require_full_window: bool,
) -> pd.Series:
    """Calculate a right-closed, non-centered trailing time-window statistic."""

    indexed = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps), dtype="float64")
    rolling = indexed.rolling(window=window, min_periods=min_periods, center=False, closed="right")
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std()
    else:
        result = rolling.sum()
    if require_full_window and len(result):
        full_at = result.index[0] + pd.Timedelta(window)
        result.loc[result.index < full_at] = np.nan
    return pd.Series(result.to_numpy(), index=values.index, dtype="float64")


def exact_time_lag(values: pd.Series, timestamps: pd.Series, lag: str) -> pd.Series:
    """Return values only for an exact t-lag timestamp; never use an as-of future value."""

    lookup = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps))
    requested = pd.DatetimeIndex(timestamps) - pd.Timedelta(lag)
    return pd.Series(lookup.reindex(requested).to_numpy(), index=values.index, dtype="float64")
