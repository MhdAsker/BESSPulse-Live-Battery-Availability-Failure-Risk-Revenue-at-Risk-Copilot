from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from besspulse import BatterySimulator, SimulationConfig
from features.battery import build_rack_features, build_site_features
from features.build import generate_feature_frames, write_feature_snapshot
from features.registry import FEATURE_REGISTRY, FEATURE_SET_VERSION
from features.validation import (
    validate_feature_frame,
    validate_no_infinite,
    validate_no_leakage,
)


def test_leakage_guard_blocks_truth_future_and_target_fields() -> None:
    for column in [
        "fault_type",
        "ground_truth_label",
        "future_fault_state",
        "failure_within_6h",
        "target_delivery_failure",
    ]:
        with pytest.raises(ValueError, match="Leakage-prohibited"):
            validate_no_leakage(["soc", column])


def test_duplicate_keys_and_infinite_values_fail_loudly() -> None:
    duplicate = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01", tz=UTC)] * 2,
            "asset_id": ["BESS-001"] * 2,
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_feature_frame(duplicate, ["asset_id"])
    with pytest.raises(ValueError, match="infinite"):
        validate_no_infinite(pd.DataFrame({"value": [1.0, np.inf]}))


def test_ground_truth_input_is_not_converted_to_features() -> None:
    simulator = BatterySimulator()
    run = simulator.simulate(datetime(2026, 1, 1, tzinfo=UTC), [2.0, 3.0])
    site = pd.DataFrame([row.model_dump() for row in run.site_telemetry])
    rack = pd.DataFrame([row.model_dump() for row in run.rack_telemetry])
    site["fault_type"] = "PCS_DERATING"
    site["ground_truth_label"] = "fault"
    rack["future_fault_state"] = "fault"
    rack["failure_within_6h"] = True
    site_features = build_site_features(site)
    rack_features = build_rack_features(rack)
    forbidden = {
        "fault_type",
        "ground_truth_label",
        "future_fault_state",
        "failure_within_6h",
        "fault_is_derating",
        "known_fault",
    }
    assert forbidden.isdisjoint(site_features.columns)
    assert forbidden.isdisjoint(rack_features.columns)


def _market_rows(start: datetime) -> pd.DataFrame:
    metrics = [
        "day_ahead_price",
        "actual_load",
        "wind_onshore_generation",
        "wind_offshore_generation",
        "solar_generation",
    ]
    values = [50.0, 60_000.0, 15_000.0, 4_000.0, 2_000.0]
    rows = []
    for offset in [0, 1]:
        timestamp = start + timedelta(hours=offset)
        rows.extend(
            {
                "timestamp_utc": timestamp,
                "metric": metric,
                "value": value + offset,
                "resolution": "PT60M",
            }
            for metric, value in zip(metrics, values, strict=True)
        )
    return pd.DataFrame(rows)


def test_end_to_end_feature_pipeline_and_snapshot_are_reproducible(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    simulator = BatterySimulator()
    run = simulator.simulate(start, [5.0] * 13)
    site = pd.DataFrame([row.model_dump() for row in run.site_telemetry])
    rack = pd.DataFrame([row.model_dump() for row in run.rack_telemetry])
    market = _market_rows(start)
    config = SimulationConfig()
    first = generate_feature_frames(site, rack, market, config=config)
    second = generate_feature_frames(site, rack, market, config=config)
    assert set(first) == {
        "site_features",
        "rack_features",
        "market_features",
        "combined_site_market_features",
    }
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    validate_feature_frame(first["site_features"], ["asset_id"])
    validate_feature_frame(first["rack_features"], ["rack_id"])
    validate_no_leakage(list(first["combined_site_market_features"].columns))
    snapshot = write_feature_snapshot(
        first,
        config=config,
        requested_start=start,
        requested_end=start + timedelta(hours=1, minutes=5),
        output_root=tmp_path,
    )
    repeated = write_feature_snapshot(
        second,
        config=config,
        requested_start=start,
        requested_end=start + timedelta(hours=1, minutes=5),
        output_root=tmp_path,
    )
    assert snapshot.created is True
    assert repeated.created is False
    assert snapshot.path == repeated.path
    assert snapshot.metadata["feature_set_version"] == FEATURE_SET_VERSION
    assert snapshot.metadata["frames"]["rack_features"]["row_count"] == 13 * 32
    assert (snapshot.path / "rack_features.parquet").exists()


def test_feature_registry_has_causal_derived_metadata() -> None:
    assert FEATURE_SET_VERSION == "v1"
    assert FEATURE_REGISTRY
    assert all(item.causal for item in FEATURE_REGISTRY.values())
    assert all(item.provenance == "DERIVED" for item in FEATURE_REGISTRY.values())
