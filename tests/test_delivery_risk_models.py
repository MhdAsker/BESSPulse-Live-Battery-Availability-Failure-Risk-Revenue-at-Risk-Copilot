from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from besspulse.database import Base, ModelPredictionModel
from models.artifact import load_model_artifact, save_model_artifact
from models.delivery_risk.calibrate import fit_calibrators
from models.delivery_risk.config import DeliveryRiskConfig
from models.delivery_risk.dataset import (
    build_risk_dataset,
    purged_chronological_split,
)
from models.delivery_risk.evaluate import classification_metrics, reliability_bins
from models.delivery_risk.explain import global_feature_importance
from models.delivery_risk.predict import predict_delivery_risk, predict_horizon
from models.delivery_risk.targets import generate_delivery_failure_targets
from models.delivery_risk.train import candidate_classifiers, train_delivery_risk
from models.storage import persist_delivery_risk_predictions


def training_data(periods: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="1h", tz=UTC)
    phase = np.arange(periods)
    requested = 10 + 5 * np.sin(phase / 8)
    failure = np.zeros(periods, dtype=bool)
    failure[np.arange(20, periods, 35)] = True
    actual = requested * np.where(failure, 0.75, 0.99)
    source = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": "BESS-001",
            "requested_power_mw": requested,
            "actual_power_mw": actual,
        }
    )
    feature = source.copy()
    feature["delivery_ratio"] = np.abs(actual / requested)
    feature["soc"] = 0.5 + 0.2 * np.sin(phase / 20)
    feature["available_power_fraction"] = np.where(failure, 0.8, 1.0)
    feature["residual_persistence_count"] = pd.Series(failure).rolling(6, min_periods=1).sum()
    feature["expected_power_residual_mw"] = actual - requested
    feature["site_max_temperature_peer_z"] = np.sin(phase / 11)
    feature["operating_mode"] = np.where(requested > 0, "DISCHARGING", "CHARGING")
    return feature, generate_delivery_failure_targets(source)


def config() -> DeliveryRiskConfig:
    return DeliveryRiskConfig(
        minimum_training_rows=50,
        minimum_positive_rows=3,
        minimum_failure_events=1,
        minimum_calibration_rows=10,
        rf_n_estimators=20,
        lgbm_n_estimators=20,
        xgb_n_estimators=20,
    )


def test_all_candidates_calibration_probability_and_reproducibility() -> None:
    feature, targets = training_data()
    dataset = build_risk_dataset(feature, targets, 6)
    first = train_delivery_risk(dataset, config())
    second = train_delivery_risk(dataset, config())
    assert set(first.validation_comparison) == {
        "prevalence_baseline",
        "engineering_baseline",
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
    }
    inference = pd.concat([first.split.test.keys, first.split.test.predictors], axis=1)
    one = predict_horizon(first.artifact, inference)
    two = predict_horizon(second.artifact, inference)
    probability = one["failure_probability_6h"]
    assert probability.between(0, 1).all()
    np.testing.assert_allclose(probability, two["failure_probability_6h"])
    assert "failure_within_6h" not in inference
    assert set(one["data_provenance"]) == {"MODEL_PREDICTION"}


def test_feature_contract_artifact_reload_and_importance(tmp_path) -> None:
    feature, targets = training_data()
    result = train_delivery_risk(build_risk_dataset(feature, targets, 6), config())
    inference = pd.concat([result.split.test.keys, result.split.test.predictors], axis=1)
    before = predict_horizon(result.artifact, inference)
    model_path, metadata_path = save_model_artifact(result.artifact, tmp_path)
    loaded = load_model_artifact(model_path)
    after = predict_horizon(loaded, inference[inference.columns[::-1]])
    np.testing.assert_allclose(before["failure_probability_6h"], after["failure_probability_6h"])
    metadata = metadata_path.read_text(encoding="utf-8")
    assert '"horizon_hours": 6' in metadata
    assert '"dataset_hash"' in metadata
    assert '"calibration_date_range"' in metadata
    assert len(global_feature_importance(loaded)) > 0
    with pytest.raises(ValueError, match="Missing model features"):
        predict_horizon(loaded, inference.drop(columns=[loaded.feature_names[0]]))


def test_known_metrics_brier_confusion_and_reliability() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(actual, probability)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc_average_precision"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.085)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert sum(entry["count"] for entry in reliability_bins(actual, probability, 4)) == 4


def test_platt_and_isotonic_use_only_supplied_held_out_values() -> None:
    probability = np.array([0.05, 0.2, 0.7, 0.9])
    actual = np.array([0, 0, 1, 1])
    calibrators = fit_calibrators(probability, actual, random_seed=42)
    assert set(calibrators) == {"identity", "platt", "isotonic"}
    for calibrator in calibrators.values():
        calibrated = calibrator.predict(probability)
        assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_delivery_preprocessor_fits_training_only_and_missing_horizon_fails() -> None:
    feature, targets = training_data()
    dataset = build_risk_dataset(feature, targets, 6)
    split = purged_chronological_split(dataset, config())
    estimator = candidate_classifiers(dataset, config())["logistic_regression"]
    estimator.fit(split.train.predictors, split.train.target.astype(int))
    numeric_columns = estimator.named_steps["preprocess"].transformers_[0][2]
    statistics = (
        estimator.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    soc_index = list(numeric_columns).index("soc")
    assert statistics[soc_index] == pytest.approx(split.train.predictors["soc"].median())
    trained = train_delivery_risk(dataset, config()).artifact
    with pytest.raises(ValueError, match="Missing delivery-risk models"):
        inference = pd.concat([split.test.keys, split.test.predictors], axis=1)
        predict_delivery_risk({6: trained}, inference)


def test_delivery_probability_storage_has_horizon_and_no_future_actual() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    frame = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01", tz=UTC)],
            "asset_id": ["BESS-001"],
            "failure_probability_6h": [0.7],
        }
    )
    with Session(engine) as session:
        written = persist_delivery_risk_predictions(
            session,
            frame,
            horizon_hours=6,
            model_name="random_forest",
            model_version="delivery_risk_6h_v1",
        )
        stored = session.scalar(select(ModelPredictionModel))
    assert written == 1
    assert stored is not None
    assert stored.prediction_horizon_hours == 6
    assert stored.predicted_class is True
    assert stored.actual_value is None
    assert stored.prediction_provenance == "MODEL_PREDICTION"
