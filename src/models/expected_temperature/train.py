"""Validation-selected expected-temperature training and walk-forward history."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from features.registry import FEATURE_SET_VERSION
from models.artifact import ModelArtifact
from models.common import (
    DatasetSplit,
    ModelDataset,
    build_candidate_estimators,
    chronological_split,
    healthy_training_rows,
    model_hyperparameters,
    regression_metrics,
    select_simplest_within_tolerance,
)
from models.expected_temperature.config import ExpectedTemperatureConfig
from models.expected_temperature.dataset import (
    TEMPERATURE_CATEGORICAL_FEATURES,
    TEMPERATURE_NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class TemperatureTrainingResult:
    artifact: ModelArtifact
    split: DatasetSplit
    validation_comparison: dict[str, dict[str, float]]
    test_metrics: dict[str, float]


def _fit_named(name: str, dataset: ModelDataset, config: ExpectedTemperatureConfig) -> Any:
    X, y = healthy_training_rows(dataset, config)
    estimator = build_candidate_estimators(
        list(TEMPERATURE_NUMERIC_FEATURES),
        list(TEMPERATURE_CATEGORICAL_FEATURES),
        "temperature_lag_5m",
        config,
    )[name]
    estimator.fit(X, y)
    return estimator


def train_expected_temperature(
    dataset: ModelDataset, config: ExpectedTemperatureConfig | None = None
) -> TemperatureTrainingResult:
    cfg = config or ExpectedTemperatureConfig()
    split = chronological_split(dataset, cfg)
    X_train, y_train = healthy_training_rows(split.train, cfg)
    validation_mask = (
        split.validation.healthy
        & split.validation.target.notna()
        & split.validation.predictors["temperature_lag_5m"].notna()
    )
    if validation_mask.sum() < 2:
        raise ValueError("Expected-temperature validation has too few healthy rows")
    X_validation = split.validation.predictors.loc[validation_mask]
    y_validation = split.validation.target.loc[validation_mask]
    candidates = build_candidate_estimators(
        list(TEMPERATURE_NUMERIC_FEATURES),
        list(TEMPERATURE_CATEGORICAL_FEATURES),
        "temperature_lag_5m",
        cfg,
    )
    comparison: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        comparison[name] = regression_metrics(y_validation, estimator.predict(X_validation))
    selected = select_simplest_within_tolerance(comparison, cfg.selection_rmse_tolerance)
    estimator = candidates[selected]
    test_prediction = estimator.predict(split.test.predictors)
    test_metrics = regression_metrics(split.test.target, test_prediction)
    artifact = ModelArtifact(
        estimator=estimator,
        feature_names=(*TEMPERATURE_NUMERIC_FEATURES, *TEMPERATURE_CATEGORICAL_FEATURES),
        numeric_features=TEMPERATURE_NUMERIC_FEATURES,
        categorical_features=TEMPERATURE_CATEGORICAL_FEATURES,
        model_name=selected,
        model_version=cfg.model_version,
        target_name=dataset.target_name,
        metadata={
            "created_at_utc": datetime.now(UTC).isoformat(),
            "feature_set_version": FEATURE_SET_VERSION,
            "dataset_hash": dataset.dataset_hash,
            "random_seed": cfg.random_seed,
            "hyperparameters": model_hyperparameters(estimator),
            "training_date_range": [
                str(split.train.keys["timestamp_utc"].min()),
                str(split.train.keys["timestamp_utc"].max()),
            ],
            "validation_date_range": [
                str(split.validation.keys["timestamp_utc"].min()),
                str(split.validation.keys["timestamp_utc"].max()),
            ],
            "test_date_range": [
                str(split.test.keys["timestamp_utc"].min()),
                str(split.test.keys["timestamp_utc"].max()),
            ],
            "validation_comparison": comparison,
            "selection_policy": (
                "simplest model within configured RMSE tolerance of validation best"
            ),
            "selection_rmse_tolerance": cfg.selection_rmse_tolerance,
            "test_metrics": test_metrics,
            "training_data_provenance": ["SIMULATED", "DERIVED"],
            "prediction_provenance": "MODEL_PREDICTION",
            "rack_identity_used_as_predictor": False,
        },
    )
    return TemperatureTrainingResult(artifact, split, comparison, test_metrics)


def walk_forward_expected_temperature(
    dataset: ModelDataset,
    *,
    model_name: str = "ridge_linear",
    config: ExpectedTemperatureConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ExpectedTemperatureConfig()
    result = dataset.keys.copy()
    result["expected_temperature_c"] = np.nan
    result["training_max_timestamp"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    timestamps = pd.DatetimeIndex(result["timestamp_utc"].unique()).sort_values()
    for block_start in range(0, len(timestamps), cfg.walk_forward_block_timestamps):
        block_times = timestamps[block_start : block_start + cfg.walk_forward_block_timestamps]
        first_time = block_times[0]
        train_positions = np.flatnonzero((dataset.keys["timestamp_utc"] < first_time).to_numpy())
        predict_positions = np.flatnonzero(
            dataset.keys["timestamp_utc"].isin(block_times).to_numpy()
        )
        if int(dataset.healthy.iloc[train_positions].sum()) < cfg.minimum_training_rows:
            continue
        train_dataset = dataset.subset(train_positions)
        estimator = _fit_named(model_name, train_dataset, cfg)
        result.loc[predict_positions, "expected_temperature_c"] = estimator.predict(
            dataset.predictors.iloc[predict_positions]
        )
        result.loc[predict_positions, "training_max_timestamp"] = train_dataset.keys[
            "timestamp_utc"
        ].max()
    result["model_version"] = cfg.model_version
    result["model_provenance"] = "MODEL_PREDICTION"
    return result


def unseen_rack_diagnostic(
    dataset: ModelDataset,
    rack_id: str,
    model_name: str,
    config: ExpectedTemperatureConfig,
) -> dict[str, float]:
    train_positions = np.flatnonzero((dataset.keys["rack_id"] != rack_id).to_numpy())
    test_positions = np.flatnonzero((dataset.keys["rack_id"] == rack_id).to_numpy())
    estimator = _fit_named(model_name, dataset.subset(train_positions), config)
    held_out = dataset.subset(test_positions)
    valid = held_out.predictors["temperature_lag_5m"].notna() & held_out.target.notna()
    return regression_metrics(
        held_out.target.loc[valid], estimator.predict(held_out.predictors.loc[valid])
    )
