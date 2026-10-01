from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from besspulse.config import FeatureConfig, SimulationConfig
from features.peer import add_peer_features


def peer_config(minimum: int = 3) -> SimulationConfig:
    return SimulationConfig(features=FeatureConfig(peer_min_population=minimum))


def peers(temperatures=None, availability=None) -> pd.DataFrame:
    temperatures = temperatures or [100.0, 24.0, 25.0, 26.0]
    availability = availability or [True, True, True, True]
    return pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01", tz=UTC)] * 4,
            "rack_id": ["R-01", "R-02", "R-03", "R-04"],
            "pcs_id": ["PCS-01"] * 4,
            "availability": availability,
            "temperature_mean_c": temperatures,
            "voltage_spread_v": [10.0, 2.0, 3.0, 4.0],
            "rte": [0.80, 0.90, 0.91, 0.92],
            "soc": [0.70, 0.50, 0.51, 0.52],
            "rack_power_residual_mw": [-1.0, -0.1, 0.0, 0.1],
        }
    )


def test_target_is_excluded_from_median_robust_zscore_and_percentile() -> None:
    result = add_peer_features(peers(), peer_config())
    target = result.loc[result["rack_id"] == "R-01"].iloc[0]
    assert target["temperature_peer_median"] == 25.0
    assert target["temperature_peer_deviation"] == 75.0
    assert target["temperature_peer_robust_zscore"] == pytest.approx(75 / 1.4826)
    assert target["temperature_peer_percentile"] == 1.0
    assert target["peer_count"] == 3
    assert target["peer_group"] == "PCS"


def test_changing_target_does_not_change_its_own_peer_baseline() -> None:
    first = add_peer_features(peers(), peer_config())
    changed = peers(temperatures=[1000.0, 24.0, 25.0, 26.0])
    second = add_peer_features(changed, peer_config())
    baseline_first = first.loc[first["rack_id"] == "R-01", "temperature_peer_median"].iloc[0]
    baseline_second = second.loc[second["rack_id"] == "R-01", "temperature_peer_median"].iloc[0]
    assert baseline_first == baseline_second == 25.0


def test_minimum_population_and_unavailable_peer_rules() -> None:
    frame = peers(availability=[True, True, True, False])
    result = add_peer_features(frame, peer_config(minimum=3))
    target = result.loc[result["rack_id"] == "R-01"].iloc[0]
    assert target["peer_count"] == 2
    assert pd.isna(target["temperature_peer_median"])
    assert pd.isna(target["peer_group"])


def test_zero_mad_is_zero_only_at_median_and_otherwise_missing() -> None:
    at_median = add_peer_features(peers([25.0, 25.0, 25.0, 25.0]), peer_config())
    assert (
        at_median.loc[at_median["rack_id"] == "R-01", "temperature_peer_robust_zscore"].iloc[0]
        == 0.0
    )
    away = add_peer_features(peers([30.0, 25.0, 25.0, 25.0]), peer_config())
    assert np.isnan(away.loc[away["rack_id"] == "R-01", "temperature_peer_robust_zscore"].iloc[0])


def test_peer_features_drop_ground_truth_and_do_not_use_it_for_eligibility() -> None:
    frame = peers()
    frame["fault_type"] = ["THERMAL_DRIFT", "NORMAL", "NORMAL", "NORMAL"]
    frame["ground_truth_label"] = ["fault", "normal", "normal", "normal"]
    result = add_peer_features(frame, peer_config())
    assert "fault_type" not in result
    assert "ground_truth_label" not in result
    assert result.loc[result["rack_id"] == "R-01", "peer_count"].iloc[0] == 3
