from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import BatteryConfig, FeatureConfig
from besspulse.ground_truth import FaultEvent, FaultType
from features.battery import build_site_features
from models.artifact import load_model_artifact, save_model_artifact
from models.delivery_risk.config import DeliveryRiskConfig
from models.delivery_risk.dataset import build_risk_dataset
from models.delivery_risk.predict import predict_horizon
from models.delivery_risk.targets import generate_delivery_failure_targets
from models.delivery_risk.train import train_delivery_risk
from models.expected_power.config import ExpectedPowerConfig
from models.expected_power.dataset import build_expected_power_dataset
from models.expected_power.train import walk_forward_expected_power


def test_simulator_to_calibrated_delivery_risk_artifact(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    config = SimulationConfig(
        battery=BatteryConfig(telemetry_interval_minutes=60),
        features=FeatureConfig(require_full_windows=False),
        simulation_seed=42,
    )
    fault_hours = [30, 80, 130, 180, 230, 280, 310]
    faults = [
        FaultEvent(
            fault_id=f"F-{hour}",
            fault_type=FaultType.POWER_TRACKING_ERROR,
            component_id="SITE",
            start_timestamp=start + timedelta(hours=hour),
            end_timestamp=start + timedelta(hours=hour + 3),
            severity=0.9,
            ground_truth_label="POWER_TRACKING_ERROR",
        )
        for hour in fault_hours
    ]
    simulator = BatterySimulator(config, faults)
    periods = 360
    requested = 4 * np.where((np.arange(periods) // 6) % 2 == 0, 1.0, -1.0)
    run = simulator.simulate(start, requested.tolist(), 20.0)
    raw = pd.DataFrame([row.model_dump() for row in run.site_telemetry])
    site = build_site_features(raw, config)
    site = site.drop(columns=["soc_volatility"])
    labels = pd.DataFrame(
        {
            "timestamp_utc": site["timestamp_utc"],
            "ground_truth_label": [
                "NORMAL" if not simulator.fault_engine.active_events(timestamp) else "FAULT"
                for timestamp in site["timestamp_utc"]
            ],
        }
    )
    expected_source = site.merge(labels, on="timestamp_utc")
    expected_config = ExpectedPowerConfig(
        minimum_training_rows=30,
        walk_forward_block_timestamps=24,
        rf_n_estimators=10,
        lgbm_n_estimators=10,
    )
    expected_dataset = build_expected_power_dataset(expected_source, expected_config)
    walk_forward = walk_forward_expected_power(expected_dataset, config=expected_config)
    site["expected_actual_power_mw"] = walk_forward["expected_actual_power_mw"]
    site["expected_power_residual_mw"] = site["actual_power_mw"] - site["expected_actual_power_mw"]
    targets = generate_delivery_failure_targets(site)
    risk_config = DeliveryRiskConfig(
        minimum_training_rows=30,
        minimum_positive_rows=2,
        minimum_failure_events=1,
        rf_n_estimators=10,
        lgbm_n_estimators=10,
        xgb_n_estimators=10,
    )
    assert {"failure_within_6h", "failure_within_12h", "failure_within_24h"}.issubset(targets)
    dataset = build_risk_dataset(site, targets, 6)
    result = train_delivery_risk(dataset, risk_config)
    path, _ = save_model_artifact(result.artifact, tmp_path)
    loaded = load_model_artifact(path)
    inference = pd.concat([result.split.test.keys, result.split.test.predictors], axis=1)
    prediction = predict_horizon(loaded, inference)
    assert prediction["failure_probability_6h"].between(0, 1).all()
    assert all("failure_within" not in column for column in loaded.feature_names)
