from datetime import UTC
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from models.delivery_risk.config import DeliveryRiskConfig
from models.delivery_risk.dataset import build_risk_dataset, purged_chronological_split
from models.delivery_risk.targets import generate_delivery_failure_targets


def telemetry(periods: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="1h", tz=UTC)
    requested = np.full(periods, 10.0)
    actual = requested.copy()
    actual[[0, 6, 13, 25, 45, 65]] = 8.0
    requested[5] = 0.05
    actual[5] = 0.0
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "requested_power_mw": requested,
            "actual_power_mw": actual,
        }
    )


def features(periods: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="1h", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "requested_power_mw": 10.0,
            "actual_power_mw": 10.0,
            "delivery_ratio": 1.0,
            "soc": np.linspace(0.2, 0.8, periods),
            "available_power_fraction": 1.0,
            "operating_mode": "DISCHARGING",
        }
    )


def test_target_interval_endpoint_idle_irregular_and_censoring() -> None:
    frame = telemetry()
    targets = generate_delivery_failure_targets(frame)
    # Event at T is excluded, but the event exactly at T+6h is included.
    assert targets.loc[0, "failure_within_6h"] == 1
    assert targets.loc[6, "failure_within_6h"] == 0
    assert targets.loc[7, "failure_within_6h"] == 1
    assert targets.loc[1, "failure_within_12h"] == 1
    assert targets.loc[1, "failure_within_24h"] == 1
    assert not bool(targets.loc[5, "delivery_failure_at_timestamp"])
    assert pd.isna(targets.iloc[-1]["failure_within_6h"])
    assert pd.isna(targets.iloc[-1]["failure_within_24h"])

    irregular = frame.drop(index=[2, 3, 4]).reset_index(drop=True)
    irregular_targets = generate_delivery_failure_targets(irregular)
    assert irregular_targets.loc[0, "failure_within_6h"] == 1


def test_targets_come_from_power_not_fault_presence() -> None:
    frame = telemetry()
    frame["fault_type"] = "PCS_DERATING"
    target = generate_delivery_failure_targets(frame)
    assert not bool(target.loc[1, "delivery_failure_at_timestamp"])
    assert bool(target.loc[6, "delivery_failure_at_timestamp"])


def test_dataset_rejects_targets_truth_and_future_fields() -> None:
    target = generate_delivery_failure_targets(telemetry(240))
    clean = features()
    dataset = build_risk_dataset(clean, target, 6)
    assert "failure_within_6h" not in dataset.predictors
    for prohibited in ["failure_within_6h", "fault_type", "ground_truth_label", "future_alarm"]:
        bad = clean.copy()
        bad[prohibited] = 0
        with pytest.raises(ValueError, match="Leakage"):
            build_risk_dataset(bad, target, 6)


def test_purged_split_is_ordered_and_target_windows_do_not_cross() -> None:
    frame = features(480)
    source = frame[["timestamp_utc", "asset_id", "requested_power_mw", "actual_power_mw"]].copy()
    source.loc[np.arange(20, 450, 30), "actual_power_mw"] = 8.0
    targets = generate_delivery_failure_targets(source)
    dataset = build_risk_dataset(frame, targets, 24)
    split = purged_chronological_split(dataset, DeliveryRiskConfig())
    partitions = [split.train, split.validation, split.calibration, split.test]
    for left, right in pairwise(partitions):
        assert left.keys.timestamp_utc.max() < right.keys.timestamp_utc.min()
        last_target_endpoint = left.keys.timestamp_utc.max() + pd.Timedelta(hours=24)
        assert last_target_endpoint < right.keys.timestamp_utc.min()
    assert all(partition.target.notna().all() for partition in partitions)
