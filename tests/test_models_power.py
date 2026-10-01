from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from models.artifact import load_model_artifact, save_model_artifact
from models.common import build_candidate_estimators, chronological_split, healthy_training_rows
from models.expected_power.config import ExpectedPowerConfig
from models.expected_power.dataset import (
    POWER_CATEGORICAL_FEATURES,
    POWER_NUMERIC_FEATURES,
    build_expected_power_dataset,
)
from models.expected_power.predict import add_power_residuals, predict_expected_power
from models.expected_power.train import train_expected_power, walk_forward_expected_power


def power_frame(rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz=UTC)
    requested = 10 * np.sin(np.arange(rows) / 8)
    actual = requested * 0.98 + 0.1 * np.sin(np.arange(rows) / 3)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "requested_power_mw": requested,
            "actual_power_mw": actual,
            "soc": 0.5 + 0.1 * np.sin(np.arange(rows) / 20),
            "ambient_temperature_c": 20 + 3 * np.sin(np.arange(rows) / 30),
            "available_racks": 32,
            "available_pcs": 4,
            "available_rack_fraction": 1.0,
            "available_pcs_fraction": 1.0,
            "available_power_mw": 20.0,
            "recent_alarm_count": 0.0,
            "alarm_frequency": 0.0,
            "rte": 0.92,
            "operating_mode": np.where(requested >= 0, "DISCHARGING", "CHARGING"),
            "ground_truth_label": np.where(np.arange(rows) < 80, "NORMAL", "FAULT"),
            "fault_type": np.where(np.arange(rows) < 80, "NORMAL", "PCS_DERATING"),
            "failure_within_6h": False,
        }
    )


def config() -> ExpectedPowerConfig:
    return ExpectedPowerConfig(
        minimum_training_rows=20,
        walk_forward_block_timestamps=10,
        rf_n_estimators=20,
        lgbm_n_estimators=25,
    )


def test_power_dataset_excludes_target_and_all_truth_fields() -> None:
    source = power_frame()
    source["data_provenance"] = "SIMULATED"
    dataset = build_expected_power_dataset(source, config())
    assert "actual_power_mw" not in dataset.predictors
    assert "fault_type" not in dataset.predictors
    assert "ground_truth_label" not in dataset.predictors
    assert "failure_within_6h" not in dataset.predictors
    assert dataset.keys["timestamp_utc"].is_monotonic_increasing
    assert set(source["data_provenance"]) == {"SIMULATED"}


def test_offline_training_requires_explicit_healthy_labels() -> None:
    with pytest.raises(ValueError, match="requires ground_truth_label"):
        build_expected_power_dataset(power_frame().drop(columns=["ground_truth_label"]), config())


def test_chronological_split_and_train_only_imputer() -> None:
    source = power_frame()
    source.loc[0, "soc"] = np.nan
    source.loc[60:, "soc"] = 100.0
    dataset = build_expected_power_dataset(source, config())
    split = chronological_split(dataset, config())
    assert split.train.keys["timestamp_utc"].max() < split.validation.keys["timestamp_utc"].min()
    assert split.validation.keys["timestamp_utc"].max() < split.test.keys["timestamp_utc"].min()
    X, y = healthy_training_rows(split.train, config())
    estimator = build_candidate_estimators(
        list(POWER_NUMERIC_FEATURES),
        list(POWER_CATEGORICAL_FEATURES),
        "requested_power_mw",
        config(),
    )["ridge_linear"]
    estimator.fit(X, y)
    statistics = (
        estimator.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    soc_index = list(POWER_NUMERIC_FEATURES).index("soc")
    assert statistics[soc_index] == pytest.approx(X["soc"].median())
    assert statistics[soc_index] < 1.0


def test_chronological_split_does_not_cut_a_known_fault_event() -> None:
    source = power_frame()
    source["fault_id"] = pd.NA
    source.loc[55:65, "fault_id"] = "EVENT-1"
    dataset = build_expected_power_dataset(source, config())
    split = chronological_split(dataset, config())
    memberships = []
    for name, partition in [
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ]:
        if partition.evaluation_metadata["fault_id"].eq("EVENT-1").any():
            memberships.append(name)
    assert memberships == ["validation"]


def test_all_power_candidates_train_and_prediction_needs_no_target() -> None:
    dataset = build_expected_power_dataset(power_frame(), config())
    result = train_expected_power(dataset, config())
    assert set(result.validation_comparison) == {
        "naive",
        "ridge_linear",
        "random_forest",
        "lightgbm",
    }
    inference = pd.concat([dataset.keys, dataset.predictors], axis=1)
    prediction = predict_expected_power(result.artifact, inference)
    assert "actual_power_mw" not in inference
    assert prediction["expected_actual_power_mw"].notna().all()
    assert set(prediction["model_provenance"]) == {"MODEL_PREDICTION"}
    invalid = inference.copy()
    invalid["soc"] = invalid["soc"].astype(object)
    invalid.loc[0, "soc"] = "not-a-number"
    with pytest.raises(ValueError, match="Incompatible numeric feature"):
        predict_expected_power(result.artifact, invalid)


def test_power_residual_and_directional_shortfall_formula() -> None:
    timestamps = pd.date_range("2026-01-01", periods=2, freq="5min", tz=UTC)
    prediction = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "expected_actual_power_mw": [10.0, -10.0],
            "model_version": "expected_power_v1",
            "model_provenance": "MODEL_PREDICTION",
        }
    )
    actual = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "actual_power_mw": [8.0, -8.0],
            "requested_power_mw": [10.0, -10.0],
        }
    )
    residual = add_power_residuals(prediction, actual)
    assert residual["power_residual_mw"].tolist() == [-2.0, 2.0]
    assert residual["delivery_shortfall_mw"].tolist() == [2.0, 2.0]
    assert set(residual["residual_provenance"]) == {"DERIVED"}


def test_power_artifact_reload_preserves_contract_and_column_order_independence(tmp_path) -> None:
    dataset = build_expected_power_dataset(power_frame(), config())
    result = train_expected_power(dataset, config())
    model_path, metadata_path = save_model_artifact(result.artifact, tmp_path)
    loaded = load_model_artifact(model_path)
    inference = pd.concat([dataset.keys, dataset.predictors], axis=1)
    original = predict_expected_power(result.artifact, inference)
    reordered = predict_expected_power(loaded, inference[inference.columns[::-1]])
    np.testing.assert_allclose(
        original["expected_actual_power_mw"], reordered["expected_actual_power_mw"]
    )
    metadata = metadata_path.read_text(encoding="utf-8")
    assert dataset.dataset_hash in metadata
    assert "expected_power_v1" in metadata
    assert '"training_date_range"' in metadata
    assert '"validation_date_range"' in metadata
    assert '"test_date_range"' in metadata
    assert tuple(loaded.feature_names) == result.artifact.feature_names


def test_power_walk_forward_is_strictly_past_and_future_invariant() -> None:
    source = power_frame()
    dataset = build_expected_power_dataset(source, config())
    baseline = walk_forward_expected_power(dataset, config=config())
    valid = baseline["expected_actual_power_mw"].notna()
    assert (~valid).any()
    assert (
        baseline.loc[valid, "training_max_timestamp"] < baseline.loc[valid, "timestamp_utc"]
    ).all()
    changed = source.copy()
    changed.loc[70:, "actual_power_mw"] = 999.0
    changed_dataset = build_expected_power_dataset(changed, config())
    recomputed = walk_forward_expected_power(changed_dataset, config=config())
    cutoff = source.loc[60, "timestamp_utc"]
    mask = baseline["timestamp_utc"] <= cutoff
    np.testing.assert_allclose(
        baseline.loc[mask, "expected_actual_power_mw"],
        recomputed.loc[mask, "expected_actual_power_mw"],
        equal_nan=True,
    )
