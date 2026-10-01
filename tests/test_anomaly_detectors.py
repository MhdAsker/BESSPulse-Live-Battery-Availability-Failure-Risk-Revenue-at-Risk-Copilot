from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.anomaly.artifact import AnomalyArtifact, load_anomaly_artifact, save_anomaly_artifact
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import build_anomaly_dataset, chronological_anomaly_split
from models.anomaly.engineering import score_engineering
from models.anomaly.isolation_forest import (
    IsolationForestArtifact,
    fit_isolation_forest,
    score_isolation_forest,
)
from models.anomaly.residual import (
    directional_delivery_shortfall,
    fit_residual_reference,
    score_residual,
)
from models.anomaly.robust import robust_zscore


def _dataset(rows: int = 60):
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "rack_id": ["RACK-01"] * rows,
            "pcs_id": ["PCS-01"] * rows,
            "temperature_vs_ambient_c": np.linspace(4, 7, rows),
            "voltage_spread_v": np.linspace(2, 3, rows),
            "expected_temperature_c": np.linspace(25, 26, rows),
            "thermal_residual_c": np.linspace(-0.2, 0.2, rows),
            "delivery_shortfall_mw": np.zeros(rows),
        }
    )
    metadata = frame[["timestamp_utc", "rack_id", "pcs_id"]].copy()
    metadata["ground_truth_label"] = "NORMAL"
    return build_anomaly_dataset(frame, metadata)


def test_zero_mad_policy_is_finite_or_missing() -> None:
    peers = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    assert robust_zscore(1.0, peers, 1e-9) == 0.0
    assert np.isnan(robust_zscore(2.0, peers, 1e-9))
    assert np.isfinite(robust_zscore(4.0, np.array([1.0, 2.0, 3.0]), 1e-9))


@pytest.mark.parametrize(
    "leakage_column",
    [
        "severity",
        "failure_within_6h",
        "failure_within_12h",
        "failure_within_24h",
        "failure_event_id",
        "time_to_failure",
        "local_shap_temperature",
    ],
)
def test_anomaly_dataset_rejects_ground_truth_predictors(leakage_column: str) -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "rack_id": ["RACK-01"],
            "pcs_id": ["PCS-01"],
            "temperature_vs_ambient_c": [5.0],
            leakage_column: [0.5],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        build_anomaly_dataset(frame)


def test_isolation_forest_is_reproducible_and_round_trips(tmp_path: Path) -> None:
    dataset = _dataset()
    config = AnomalyConfig(minimum_training_rows=20, iforest_n_estimators=20)
    first = fit_isolation_forest(dataset, config)
    second = fit_isolation_forest(dataset, config)
    first_scores = score_isolation_forest(dataset, first)["anomaly_score"]
    second_scores = score_isolation_forest(dataset, second)["anomaly_score"]
    np.testing.assert_allclose(first_scores, second_scores)
    wrapped = AnomalyArtifact(
        "isolation_forest",
        first.version,
        first,
        first.feature_names,
        0.1,
        2,
        {"seed": 42},
    )
    model_path, _ = save_anomaly_artifact(wrapped, tmp_path)
    loaded = load_anomaly_artifact(model_path)
    loaded_scores = score_isolation_forest(dataset, loaded.fitted_object)["anomaly_score"]
    np.testing.assert_allclose(first_scores, loaded_scores)


def test_residual_reference_is_train_only_and_high_is_anomalous() -> None:
    train = _dataset()
    reference = fit_residual_reference(train, AnomalyConfig(minimum_training_rows=20))
    shifted = _dataset()
    shifted.predictors.loc[:, "thermal_residual_c"] = 10.0
    scores = score_residual(shifted, reference)
    assert scores["anomaly_score"].median() > 1.0
    assert reference.training_end == str(train.keys.timestamp_utc.max())


def test_delivery_shortfall_respects_charge_and_discharge_signs() -> None:
    requested = pd.Series([2.0, -2.0, 2.0, -2.0, 0.0])
    actual = pd.Series([1.0, -1.0, 2.5, -2.5, 0.2])
    assert directional_delivery_shortfall(requested, actual).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_engineering_rule_retains_evidence_and_nonbreach() -> None:
    dataset = _dataset(3)
    dataset.predictors.loc[:, "temperature_vs_ambient_c"] = [5.0, 20.0, 5.0]
    output = score_engineering(dataset)
    assert output.loc[0, "triggered_rules"] == []
    assert output.loc[1, "triggered_rules"] == ["temperature_vs_ambient_c"]
    assert output.loc[1, "threshold_origins"] == ["PROJECT ENGINEERING ASSUMPTION"]
    assert output["data_provenance"].eq("DERIVED").all()


def test_chronological_split_is_ordered_and_identity_is_not_a_predictor() -> None:
    split = chronological_anomaly_split(_dataset(), AnomalyConfig())
    assert split.train.keys.timestamp_utc.max() < split.validation.keys.timestamp_utc.min()
    assert split.validation.keys.timestamp_utc.max() < split.test.keys.timestamp_utc.min()
    assert "rack_id" not in split.train.predictors
    assert "pcs_id" not in split.train.predictors


class _DecisionFunctionStub:
    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([0.2, -0.4])


def test_isolation_forest_project_score_negates_decision_function() -> None:
    dataset = _dataset(2)
    artifact = IsolationForestArtifact(
        _DecisionFunctionStub(),
        ("temperature_vs_ambient_c",),
        "test",
        dataset.dataset_hash,
        "start",
        "end",
        42,
    )
    output = score_isolation_forest(dataset, artifact)
    assert output["anomaly_score"].tolist() == [-0.2, 0.4]
    assert output["data_provenance"].eq("MODEL_PREDICTION / ANOMALY MODEL OUTPUT").all()
