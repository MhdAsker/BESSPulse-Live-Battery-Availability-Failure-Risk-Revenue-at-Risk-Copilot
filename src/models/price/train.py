"""Validation-only price-model selection and one-time held-out evaluation."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from models.artifact import ModelArtifact
from models.price.baselines import BlendBaseline, HourOfWeekBaseline, LagBaseline
from models.price.config import PriceModelConfig
from models.price.dataset import PriceDataset, PriceSplit, chronological_price_split
from models.price.evaluate import price_metrics, regime_metrics


def lightgbm_regressor(
    config: PriceModelConfig, *, objective: str = "regression", alpha: float | None = None
) -> LGBMRegressor:
    parameters: dict[str, Any] = {
        "objective": objective,
        "n_estimators": config.lgbm_n_estimators,
        "learning_rate": config.lgbm_learning_rate,
        "num_leaves": config.lgbm_num_leaves,
        "max_depth": config.lgbm_max_depth,
        "min_child_samples": config.lgbm_min_child_samples,
        "subsample": config.lgbm_subsample,
        "colsample_bytree": config.lgbm_colsample_bytree,
        "reg_alpha": config.lgbm_reg_alpha,
        "reg_lambda": config.lgbm_reg_lambda,
        "random_state": config.random_seed,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        parameters["alpha"] = alpha
    return LGBMRegressor(**parameters)


def candidate_estimators(config: PriceModelConfig) -> dict[str, Any]:
    return {
        "lag_24h": LagBaseline("price_lag_24h"),
        "lag_168h": LagBaseline("price_lag_168h"),
        "hour_of_week": HourOfWeekBaseline(),
        "simple_blend": BlendBaseline(),
        "lightgbm": lightgbm_regressor(config),
    }


def candidate_feature_names(model_name: str, available: list[str]) -> list[str]:
    required = {
        "lag_24h": ["price_lag_24h"],
        "lag_168h": ["price_lag_168h"],
        "hour_of_week": ["hour_of_week"],
        "simple_blend": ["price_lag_24h", "price_lag_168h", "hour_of_week"],
        "lightgbm": available,
    }
    return required[model_name]


def select_price_model(comparison: dict[str, dict[str, float]], tolerance_pct: float) -> str:
    best_mae = min(metrics["mae"] for metrics in comparison.values())
    limit = best_mae * (1 + tolerance_pct / 100)
    simplicity = ["lag_24h", "lag_168h", "hour_of_week", "simple_blend", "lightgbm"]
    return next(name for name in simplicity if comparison[name]["mae"] <= limit)


@dataclass(frozen=True)
class PriceTrainingResult:
    artifact: ModelArtifact
    split: PriceSplit
    validation_comparison: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    test_regime_metrics: dict[str, object]
    high_price_threshold: float


def train_price_model(
    dataset: PriceDataset, config: PriceModelConfig | None = None
) -> PriceTrainingResult:
    cfg = config or PriceModelConfig()
    split = chronological_price_split(dataset, cfg)
    fitted: dict[str, Any] = {}
    comparison: dict[str, dict[str, float]] = {}
    for name, estimator in candidate_estimators(cfg).items():
        features = candidate_feature_names(name, list(dataset.predictors.columns))
        estimator.fit(split.train.predictors[features], split.train.target)
        comparison[name] = price_metrics(
            split.validation.target,
            np.asarray(estimator.predict(split.validation.predictors[features]), dtype=float),
        )
        fitted[name] = estimator
    selected = select_price_model(comparison, cfg.selection_tolerance_pct)
    # Refit after selection on train + validation; test remains untouched until this point.
    train_validation_X = pd.concat(
        [split.train.predictors, split.validation.predictors], ignore_index=True
    )
    train_validation_y = pd.concat([split.train.target, split.validation.target], ignore_index=True)
    estimator = candidate_estimators(cfg)[selected]
    selected_features = candidate_feature_names(selected, list(dataset.predictors.columns))
    estimator.fit(train_validation_X[selected_features], train_validation_y)
    test_prediction = np.asarray(
        estimator.predict(split.test.predictors[selected_features]), dtype=float
    )
    test_metrics = price_metrics(split.test.target, test_prediction)
    high_threshold = float(split.train.target.quantile(cfg.high_price_training_quantile))
    regime = regime_metrics(split.test.target, test_prediction, high_threshold)
    resolution = pd.Series(dataset.timestamps.sort_values()).diff().dropna().mode().iloc[0]
    artifact = ModelArtifact(
        estimator=estimator,
        feature_names=tuple(selected_features),
        numeric_features=tuple(selected_features),
        categorical_features=(),
        model_name=selected,
        model_version=cfg.model_version,
        target_name=cfg.target_name,
        metadata={
            "created_at_utc": datetime.now(UTC).isoformat(),
            "feature_set_version": cfg.feature_set_version,
            "dataset_hash": dataset.dataset_hash,
            "market_region": cfg.market_region,
            "market_timezone": cfg.market_timezone,
            "forecast_horizon_hours": cfg.forecast_horizon_hours,
            "resolution_minutes": int(pd.Timedelta(resolution).total_seconds() / 60),
            "random_seed": cfg.random_seed,
            "selection_policy": (
                "lowest validation MAE; prefer first simpler candidate within "
                f"{cfg.selection_tolerance_pct:.1f}%"
            ),
            "validation_comparison": comparison,
            "test_metrics": test_metrics,
            "negative_and_high_price_test_metrics": regime,
            "high_price_threshold_training_quantile": cfg.high_price_training_quantile,
            "high_price_threshold_eur_per_mwh": high_threshold,
            "training_interval": [
                str(split.train.timestamps.min()),
                str(split.train.timestamps.max()),
            ],
            "validation_interval": [
                str(split.validation.timestamps.min()),
                str(split.validation.timestamps.max()),
            ],
            "test_interval": [str(split.test.timestamps.min()), str(split.test.timestamps.max())],
            "prediction_provenance": "MODEL_PREDICTION",
            "target_provenance": "REAL ENTSO-E DATA",
            "hyperparameters": estimator.get_params(deep=False),
        },
    )
    return PriceTrainingResult(artifact, split, comparison, test_metrics, regime, high_threshold)
