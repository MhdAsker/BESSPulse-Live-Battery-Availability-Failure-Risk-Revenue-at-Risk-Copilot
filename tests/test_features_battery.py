from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from besspulse.config import FeatureConfig, SimulationConfig
from features.battery import build_rack_features, build_site_features, summarize_site_kpis


def feature_config(**updates) -> SimulationConfig:
    values = {
        "power_residual_window": "15min",
        "soc_volatility_window": "15min",
        "soc_exposure_window": "15min",
        "temperature_window": "15min",
        "thermal_exposure_window": "15min",
        "rte_trend_window": "15min",
        "availability_window": "15min",
        "alarm_window": "15min",
        "high_soc_threshold": 0.75,
        "low_soc_threshold": 0.55,
        "require_full_windows": False,
    }
    values.update(updates)
    return SimulationConfig(features=FeatureConfig(**values))


def site_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="5min", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "site_requested_power_mw": [0.05, 10.0, 10.0, -10.0],
            "site_actual_power_mw": [0.05, 8.0, 9.0, -8.0],
            "available_power_mw": [20.0, 20.0, 18.0, 18.0],
            "available_energy_mwh": [20.0, 19.0, 18.0, 19.0],
            "site_soc": [0.50, 0.60, 0.80, 0.90],
            "rte": [0.92, 0.91, 0.90, 0.89],
            "available_racks": [32, 32, 31, 31],
            "available_pcs": [4, 4, 4, 4],
            "operating_mode": ["IDLE", "DISCHARGING", "DISCHARGING", "CHARGING"],
            "alarm_count": [0, 0, 1, 0],
        }
    )


def rack_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="5min", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "rack_id": ["R-01"] * 4,
            "pcs_id": ["PCS-01"] * 4,
            "soc": [0.50, 0.55, 0.60, 0.65],
            "soh_proxy": [1.0] * 4,
            "voltage_v": [1200.0, 1201.0, 1202.0, 1203.0],
            "current_a": [0.0, 10.0, 20.0, 30.0],
            "temperature_mean_c": [25.0, 26.0, 36.0, 37.0],
            "temperature_spread_c": [1.0, 1.1, 1.2, 1.3],
            "voltage_spread_v": [2.0, 2.5, 3.0, 3.5],
            "requested_power_mw": [0.0, 0.5, 0.5, -0.5],
            "actual_power_mw": [0.0, 0.4, 0.3, -0.4],
            "cumulative_throughput_mwh": [0.0, 0.1, 0.2, 0.3],
            "equivalent_full_cycles": [0.0, 0.01, 0.02, 0.03],
            "rte": [0.92, 0.91, 0.90, 0.89],
            "availability": [True, True, False, True],
            "alarm_code": [None, None, "HOT", None],
            "operating_state": ["IDLE", "DISCHARGING", "DISCHARGING", "CHARGING"],
            "ambient_temperature_c": [20.0] * 4,
        }
    )


def test_power_delivery_residual_near_zero_and_ramps() -> None:
    features = build_site_features(site_frame(), feature_config())
    assert np.isnan(features.loc[0, "delivery_ratio"])
    assert features.loc[1, "delivery_ratio"] == pytest.approx(0.8)
    assert features.loc[1, "power_residual_mw"] == -2.0
    assert features.loc[3, "power_residual_mw"] == 2.0
    assert features.loc[1, "requested_power_ramp_mw"] == pytest.approx(9.95)
    assert features.loc[2, "actual_power_ramp_mw"] == 1.0


def test_site_rolling_power_and_soc_features_are_backward_looking() -> None:
    features = build_site_features(site_frame(), feature_config())
    # The right-closed 15-minute window is (t-15m, t], so the boundary row is excluded.
    assert features.loc[3, "rolling_power_residual_mean"] == pytest.approx(-1 / 3)
    assert features.loc[1, "soc_change"] == pytest.approx(0.1)
    assert features.loc[1, "soc_change_rate"] == pytest.approx(1.2)
    assert features.loc[3, "soc_volatility"] == pytest.approx(pd.Series([0.6, 0.8, 0.9]).std())
    assert features.loc[3, "time_high_soc"] == pytest.approx(2 / 3)
    assert features.loc[2, "time_low_soc"] == pytest.approx(1 / 3)


def test_rack_thermal_electrical_performance_and_alarm_features() -> None:
    features = build_rack_features(rack_frame(), feature_config())
    assert features.loc[1, "temperature_vs_ambient_c"] == 6.0
    assert features.loc[2, "temperature_ramp_c"] == 10.0
    assert features.loc[3, "thermal_exposure_above_threshold"] == pytest.approx(2 / 3)
    assert features.loc[1, "voltage_spread_change"] == 0.5
    assert features.loc[1, "current_change"] == 10.0
    assert features.loc[3, "rte_trend"] == pytest.approx((0.89 - 0.92) / 0.25)
    assert features.loc[3, "availability_rate"] == pytest.approx(2 / 3)
    assert features.loc[3, "alarm_frequency"] == pytest.approx(1 / 3)
    assert features.loc[2, "time_since_last_alarm_minutes"] == 0.0


def test_operating_modes_are_one_hot_not_ordinal() -> None:
    features = build_site_features(site_frame(), feature_config())
    assert features.loc[0, "is_idle"]
    assert features.loc[1, "is_discharging"]
    assert features.loc[3, "is_charging"]


def test_site_kpi_summary_is_observed_not_predictive() -> None:
    site = build_site_features(site_frame(), feature_config())
    rack = build_rack_features(rack_frame(), feature_config())
    summary = summarize_site_kpis(site, feature_config(), rack)
    assert summary["rated_power_mw"] == 20.0
    assert summary["requested_power_availability"] == pytest.approx(0.0)
    assert summary["energy_throughput_mwh"] == pytest.approx(0.3)
    assert summary["data_provenance"] == "DERIVED"


def test_future_battery_changes_cannot_change_past_features() -> None:
    source = site_frame()
    baseline = build_site_features(source, feature_config())
    changed = source.copy()
    changed.loc[3, "site_actual_power_mw"] = 1000.0
    recomputed = build_site_features(changed, feature_config())
    pd.testing.assert_frame_equal(
        baseline.loc[:2], recomputed.loc[:2], check_exact=False, rtol=1e-12
    )
