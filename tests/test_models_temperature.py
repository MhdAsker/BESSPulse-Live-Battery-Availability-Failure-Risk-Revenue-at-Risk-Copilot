from datetime import UTC

import numpy as np
import pandas as pd

from models.artifact import load_model_artifact, save_model_artifact
from models.expected_temperature.config import ExpectedTemperatureConfig
from models.expected_temperature.dataset import build_expected_temperature_dataset
from models.expected_temperature.predict import (
    add_thermal_residuals,
    predict_expected_temperature,
)
from models.expected_temperature.train import (
    train_expected_temperature,
    unseen_rack_diagnostic,
    walk_forward_expected_temperature,
)


def temperature_frame(periods: int = 30, racks: int = 4) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="5min", tz=UTC)
    for rack in range(racks):
        previous = 22.0 + rack * 0.1
        for index, timestamp in enumerate(timestamps):
            power = 0.5 * np.sin(index / 4)
            ambient = 20 + 2 * np.sin(index / 10)
            temperature = previous + 0.15 * abs(power) + 0.05 * (ambient - previous)
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "rack_id": f"R-{rack:02d}",
                    "pcs_id": f"PCS-{rack // 2:02d}",
                    "temperature_mean_c": temperature,
                    "actual_power_mw": power,
                    "requested_power_mw": power,
                    "ambient_temperature_c": ambient,
                    "soc": 0.5 + 0.05 * np.sin(index / 8),
                    "throughput_change_mwh": abs(power) * 5 / 60,
                    "availability": True,
                    "operating_state": "DISCHARGING" if power >= 0 else "CHARGING",
                    "ground_truth_label": "NORMAL" if index < 24 else "THERMAL_FAULT",
                    "fault_type": "NORMAL" if index < 24 else "THERMAL_DRIFT",
                }
            )
            previous = temperature
    return pd.DataFrame(rows)


def config() -> ExpectedTemperatureConfig:
    return ExpectedTemperatureConfig(
        minimum_training_rows=20,
        walk_forward_block_timestamps=5,
        rf_n_estimators=20,
        lgbm_n_estimators=25,
    )


def test_temperature_dataset_excludes_target_truth_and_rack_identity() -> None:
    dataset = build_expected_temperature_dataset(temperature_frame(), config())
    assert "temperature_mean_c" not in dataset.predictors
    assert "fault_type" not in dataset.predictors
    assert "ground_truth_label" not in dataset.predictors
    assert "rack_id" not in dataset.predictors
    assert "pcs_id" not in dataset.predictors
    assert dataset.keys.duplicated(["timestamp_utc", "rack_id"]).sum() == 0


def test_temperature_candidates_and_lag_baseline_train_reproducibly() -> None:
    dataset = build_expected_temperature_dataset(temperature_frame(), config())
    first = train_expected_temperature(dataset, config())
    second = train_expected_temperature(dataset, config())
    assert set(first.validation_comparison) == {
        "naive",
        "ridge_linear",
        "random_forest",
        "lightgbm",
    }
    inference = pd.concat([dataset.keys, dataset.predictors], axis=1)
    one = predict_expected_temperature(first.artifact, inference)
    two = predict_expected_temperature(second.artifact, inference)
    np.testing.assert_allclose(one["expected_temperature_c"], two["expected_temperature_c"])
    assert "temperature_mean_c" not in inference


def test_thermal_residual_formula_and_provenance() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz=UTC)
    predictions = pd.DataFrame(
        {
            "timestamp_utc": [timestamp],
            "rack_id": ["UNSEEN-RACK"],
            "pcs_id": ["PCS-X"],
            "expected_temperature_c": [25.0],
            "model_version": ["expected_temperature_v1"],
            "model_provenance": ["MODEL_PREDICTION"],
        }
    )
    actual = pd.DataFrame(
        {"timestamp_utc": [timestamp], "rack_id": ["UNSEEN-RACK"], "temperature_mean_c": [27.0]}
    )
    result = add_thermal_residuals(predictions, actual)
    assert result.loc[0, "thermal_residual_c"] == 2.0
    assert result.loc[0, "absolute_thermal_residual_c"] == 2.0
    assert result.loc[0, "prediction_provenance"] == "MODEL_PREDICTION"
    assert result.loc[0, "residual_provenance"] == "DERIVED"


def test_unseen_rack_prediction_and_artifact_reload(tmp_path) -> None:
    dataset = build_expected_temperature_dataset(temperature_frame(), config())
    result = train_expected_temperature(dataset, config())
    diagnostic = unseen_rack_diagnostic(dataset, "R-03", result.artifact.model_name, config())
    assert np.isfinite(diagnostic["rmse"])
    path, _ = save_model_artifact(result.artifact, tmp_path)
    loaded = load_model_artifact(path)
    inference = pd.concat([dataset.keys, dataset.predictors], axis=1)
    inference["rack_id"] = "NEVER-SEEN"
    prediction = predict_expected_temperature(loaded, inference)
    assert prediction["expected_temperature_c"].notna().all()


def test_temperature_walk_forward_is_strictly_past_and_future_invariant() -> None:
    source = temperature_frame()
    dataset = build_expected_temperature_dataset(source, config())
    baseline = walk_forward_expected_temperature(dataset, config=config())
    valid = baseline["expected_temperature_c"].notna()
    assert (~valid).any()
    assert (
        baseline.loc[valid, "training_max_timestamp"] < baseline.loc[valid, "timestamp_utc"]
    ).all()
    changed = source.copy()
    future = changed["timestamp_utc"] >= changed["timestamp_utc"].sort_values().unique()[24]
    changed.loc[future, "temperature_mean_c"] = 100.0
    recomputed = walk_forward_expected_temperature(
        build_expected_temperature_dataset(changed, config()), config=config()
    )
    cutoff = source["timestamp_utc"].sort_values().unique()[20]
    mask = baseline["timestamp_utc"] <= cutoff
    np.testing.assert_allclose(
        baseline.loc[mask, "expected_temperature_c"],
        recomputed.loc[mask, "expected_temperature_c"],
        equal_nan=True,
    )
