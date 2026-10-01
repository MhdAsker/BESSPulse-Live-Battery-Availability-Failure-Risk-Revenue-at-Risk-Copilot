from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import FeatureConfig
from besspulse.ground_truth import FaultEvent, FaultType
from features.battery import build_rack_features
from features.peer import add_peer_features
from models.anomaly.artifact import AnomalyArtifact, load_anomaly_artifact, save_anomaly_artifact
from models.anomaly.common import apply_consecutive_persistence
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import build_anomaly_dataset, chronological_anomaly_split
from models.anomaly.evaluate import detector_metrics, select_threshold_persistence
from models.anomaly.isolation_forest import fit_isolation_forest, score_isolation_forest
from models.anomaly.residual import directional_delivery_shortfall


def test_anomaly_pipeline_is_deterministic_end_to_end(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    faults = [
        FaultEvent(
            fault_id="VAL-THERMAL",
            fault_type=FaultType.THERMAL_DRIFT,
            component_id="PCS-01-RACK-01",
            start_timestamp=start + timedelta(hours=5),
            end_timestamp=start + timedelta(hours=5, minutes=45),
            severity=0.5,
            progression_rate=0.0,
            ground_truth_label="THERMAL_DRIFT",
        ),
        FaultEvent(
            fault_id="TEST-VOLTAGE",
            fault_type=FaultType.VOLTAGE_IMBALANCE,
            component_id="PCS-01-RACK-02",
            start_timestamp=start + timedelta(hours=7),
            end_timestamp=start + timedelta(hours=7, minutes=45),
            severity=0.8,
            progression_rate=0.0,
            ground_truth_label="VOLTAGE_IMBALANCE",
        ),
    ]
    simulation_config = SimulationConfig(
        simulation_seed=42, features=FeatureConfig(require_full_windows=False)
    )
    simulator = BatterySimulator(simulation_config, faults)
    count = 100
    requested = np.tile([1.5, -1.5], count // 2).tolist()
    ambient = (20 + np.sin(np.arange(count) / 12)).tolist()
    result = simulator.simulate(start, requested, ambient)
    raw = pd.DataFrame([row.model_dump() for row in result.rack_telemetry])
    raw = raw.loc[raw["pcs_id"].eq("PCS-01")].copy()
    ambient_frame = pd.DataFrame(
        {
            "timestamp_utc": [row.timestamp_utc for row in result.site_telemetry],
            "ambient_temperature_c": [row.ambient_temperature_c for row in result.site_telemetry],
        }
    )
    raw = raw.merge(ambient_frame, on="timestamp_utc", validate="many_to_one")
    feature_frame = add_peer_features(
        build_rack_features(raw, simulation_config), simulation_config
    )
    feature_frame["expected_temperature_c"] = feature_frame.groupby("rack_id")[
        "temperature_mean_c"
    ].shift(1)
    feature_frame["thermal_residual_c"] = (
        feature_frame["temperature_mean_c"] - feature_frame["expected_temperature_c"]
    )
    feature_frame["thermal_residual_rolling_mean"] = feature_frame.groupby("rack_id")[
        "thermal_residual_c"
    ].transform(lambda series: series.rolling(3, min_periods=1).mean())
    feature_frame["delivery_shortfall_mw"] = directional_delivery_shortfall(
        feature_frame["requested_power_mw"], feature_frame["actual_power_mw"]
    )
    truth_rows: list[dict[str, object]] = []
    for row in feature_frame.itertuples():
        active = [
            event
            for event in simulator.fault_engine.active_events(row.timestamp_utc)
            if event.component_id == row.rack_id
        ]
        truth_rows.append(
            {
                "timestamp_utc": row.timestamp_utc,
                "rack_id": row.rack_id,
                "pcs_id": row.pcs_id,
                "ground_truth_label": active[0].ground_truth_label if active else "NORMAL",
                "fault_id": active[0].fault_id if active else None,
            }
        )
    dataset = build_anomaly_dataset(feature_frame, pd.DataFrame(truth_rows))
    config = AnomalyConfig(
        minimum_training_rows=20,
        iforest_n_estimators=20,
        persistence_candidates=(1, 2),
    )
    split = chronological_anomaly_split(dataset, config)
    model = fit_isolation_forest(split.train, config)
    validation_faults = pd.DataFrame(
        [
            {
                "fault_id": "VAL-THERMAL",
                "component_id": "PCS-01-RACK-01",
                "start_timestamp": pd.Timestamp(faults[0].start_timestamp),
                "end_timestamp": pd.Timestamp(faults[0].end_timestamp),
            }
        ]
    )
    validation_scores = score_isolation_forest(split.validation, model)
    thresholds = tuple(float(validation_scores["anomaly_score"].quantile(q)) for q in (0.8, 0.9))
    threshold, persistence, _ = select_threshold_persistence(
        validation_scores, split.validation.keys, validation_faults, thresholds, config
    )
    test_intervals = apply_consecutive_persistence(
        score_isolation_forest(split.test, model),
        threshold=threshold,
        intervals=persistence,
    )
    test_faults = pd.DataFrame(
        [
            {
                "fault_id": "TEST-VOLTAGE",
                "component_id": "PCS-01-RACK-02",
                "start_timestamp": pd.Timestamp(faults[1].start_timestamp),
                "end_timestamp": pd.Timestamp(faults[1].end_timestamp),
            }
        ]
    )
    metrics, matches, _ = detector_metrics(test_intervals, split.test.keys, test_faults, config)
    assert metrics["fault_event_count"] == 1
    assert metrics["false_alerts_per_day"] >= 0
    assert len(matches) == 1
    artifact = AnomalyArtifact(
        "isolation_forest",
        model.version,
        model,
        model.feature_names,
        threshold,
        persistence,
        {"dataset_hash": dataset.dataset_hash, "seed": 42},
    )
    model_path, _ = save_anomaly_artifact(artifact, tmp_path)
    reloaded = load_anomaly_artifact(model_path)
    repeated = score_isolation_forest(split.test, reloaded.fitted_object)
    aligned = test_intervals.merge(
        repeated[["timestamp_utc", "component_id", "anomaly_score"]],
        on=["timestamp_utc", "component_id"],
        suffixes=("_original", "_reloaded"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        aligned["anomaly_score_original"], aligned["anomaly_score_reloaded"], equal_nan=True
    )
