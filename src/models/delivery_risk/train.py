"""Candidate training, validation-only selection, and held-out calibration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features.registry import FEATURE_SET_VERSION
from models.artifact import ModelArtifact
from models.common import model_hyperparameters
from models.delivery_risk.artifact import CalibratedRiskEstimator
from models.delivery_risk.calibrate import fit_calibrators
from models.delivery_risk.config import DeliveryRiskConfig
from models.delivery_risk.dataset import (
    PurgedRiskSplit,
    RiskDataset,
    numeric_and_categorical,
    purged_chronological_split,
)
from models.delivery_risk.evaluate import classification_metrics


class ConstantPrevalenceClassifier:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantPrevalenceClassifier":
        self.prevalence_ = float(y.mean())
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        positive = np.full(len(X), self.prevalence_, dtype=float)
        return np.column_stack([1 - positive, positive])


class EngineeringRiskClassifier:
    """Fixed observable score; no future outcomes or fitted thresholds."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EngineeringRiskClassifier":
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        def values(column: str, default: float) -> pd.Series:
            if column not in X:
                return pd.Series(default, index=X.index, dtype=float)
            return pd.to_numeric(X[column], errors="coerce").fillna(default)

        availability = values("available_power_fraction", 1.0)
        ratio = values("delivery_ratio", 1.0)
        persistence = values("residual_persistence_count", 0.0)
        score = 0.5 * (1 - availability).clip(0, 1) + 0.4 * (1 - ratio).clip(0, 1)
        score += 0.1 * (persistence / 12).clip(0, 1)
        positive = score.clip(0, 1).to_numpy(dtype=float)
        return np.column_stack([1 - positive, positive])


def _preprocessor(numeric: list[str], categorical: list[str], *, scale: bool) -> Any:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def candidate_classifiers(dataset: RiskDataset, config: DeliveryRiskConfig) -> dict[str, Any]:
    numeric, categorical = numeric_and_categorical(dataset)
    return {
        "prevalence_baseline": ConstantPrevalenceClassifier(),
        "engineering_baseline": EngineeringRiskClassifier(),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=True)),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.logistic_c,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=config.rf_n_estimators,
                        max_depth=config.rf_max_depth,
                        min_samples_leaf=config.rf_min_samples_leaf,
                        class_weight="balanced",
                        random_state=config.random_seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=False)),
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=config.lgbm_n_estimators,
                        learning_rate=config.lgbm_learning_rate,
                        num_leaves=config.lgbm_num_leaves,
                        class_weight="balanced",
                        random_state=config.random_seed,
                        n_jobs=1,
                        deterministic=True,
                        force_col_wise=True,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=config.xgb_n_estimators,
                        learning_rate=config.xgb_learning_rate,
                        max_depth=config.xgb_max_depth,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=config.random_seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _validate_training(split: PurgedRiskSplit, config: DeliveryRiskConfig) -> None:
    y = split.train.target.astype(int)
    positives = int(y.sum())
    if len(y) < config.minimum_training_rows:
        raise ValueError(f"Insufficient training rows: {len(y)} < {config.minimum_training_rows}")
    if positives < config.minimum_positive_rows:
        raise ValueError(
            f"Insufficient positive training rows: {positives} < {config.minimum_positive_rows}"
        )
    if y.nunique() < 2:
        raise ValueError("Training target must contain both classes")
    event_column = f"future_failure_event_ids_{split.train.horizon_hours}h"
    if event_column in split.train.evaluation_metadata:
        events = {
            event for values in split.train.evaluation_metadata[event_column] for event in values
        }
        if len(events) < config.minimum_failure_events:
            raise ValueError(
                "Insufficient independent training failure events: "
                f"{len(events)} < {config.minimum_failure_events}"
            )
    for name, partition in [
        ("validation", split.validation),
        ("calibration", split.calibration),
        ("test", split.test),
    ]:
        if partition.target.astype(int).nunique() < 2:
            raise ValueError(f"{name} target must contain both classes")
    if len(split.calibration.target) < config.minimum_calibration_rows:
        raise ValueError(
            "Insufficient calibration rows: "
            f"{len(split.calibration.target)} < {config.minimum_calibration_rows}"
        )


def _select(comparison: dict[str, dict[str, Any]], tolerance: float) -> str:
    candidates = [
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
    ]
    best = max(float(comparison[name]["pr_auc_average_precision"]) for name in candidates)
    eligible = [
        name
        for name in candidates
        if float(comparison[name]["pr_auc_average_precision"]) >= best - tolerance
    ]
    return min(
        eligible, key=lambda name: (float(comparison[name]["brier_score"]), candidates.index(name))
    )


@dataclass(frozen=True)
class DeliveryRiskTrainingResult:
    artifact: ModelArtifact
    split: PurgedRiskSplit
    validation_comparison: dict[str, dict[str, Any]]
    calibration_comparison: dict[str, dict[str, Any]]
    test_metrics: dict[str, Any]


def train_delivery_risk(
    dataset: RiskDataset, config: DeliveryRiskConfig | None = None
) -> DeliveryRiskTrainingResult:
    cfg = config or DeliveryRiskConfig()
    split = purged_chronological_split(dataset, cfg)
    _validate_training(split, cfg)
    y_train = split.train.target.astype(int)
    y_validation = split.validation.target.astype(int)
    fitted: dict[str, Any] = {}
    comparison: dict[str, dict[str, Any]] = {}
    training_events = 0
    event_column = f"future_failure_event_ids_{dataset.horizon_hours}h"
    if event_column in split.train.evaluation_metadata:
        training_events = len(
            {event for values in split.train.evaluation_metadata[event_column] for event in values}
        )
    for name, estimator in candidate_classifiers(dataset, cfg).items():
        if name == "xgboost":
            negatives = int((y_train == 0).sum())
            positives = int((y_train == 1).sum())
            estimator.set_params(classifier__scale_pos_weight=negatives / positives)
        estimator.fit(split.train.predictors, y_train)
        probability = estimator.predict_proba(split.validation.predictors)[:, 1]
        comparison[name] = classification_metrics(
            y_validation,
            probability,
            threshold=cfg.classification_threshold,
            calibration_bins=cfg.calibration_bins,
        )
        fitted[name] = estimator
    selected = _select(comparison, cfg.selection_pr_auc_tolerance)
    base = fitted[selected]
    calibration_probability = base.predict_proba(split.calibration.predictors)[:, 1]
    calibrators = fit_calibrators(
        calibration_probability,
        split.calibration.target.astype(int).to_numpy(),
        random_seed=cfg.random_seed,
    )
    calibration_comparison = {
        name: classification_metrics(
            split.calibration.target.astype(int),
            calibrator.predict(calibration_probability),
            threshold=cfg.classification_threshold,
            calibration_bins=cfg.calibration_bins,
        )
        for name, calibrator in calibrators.items()
    }
    calibration_method = min(
        calibration_comparison,
        key=lambda name: float(calibration_comparison[name]["brier_score"]),
    )
    estimator = CalibratedRiskEstimator(base, calibrators[calibration_method])
    test_probability = estimator.predict_proba(split.test.predictors)[:, 1]
    test_metrics = classification_metrics(
        split.test.target.astype(int),
        test_probability,
        threshold=cfg.classification_threshold,
        calibration_bins=cfg.calibration_bins,
    )
    numeric, categorical = numeric_and_categorical(dataset)
    horizon = dataset.horizon_hours
    artifact = ModelArtifact(
        estimator=estimator,
        feature_names=tuple(dataset.predictors.columns),
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        model_name=selected,
        model_version=f"delivery_risk_{horizon}h_v1",
        target_name=f"failure_within_{horizon}h",
        metadata={
            "created_at_utc": datetime.now(UTC).isoformat(),
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_manifest": list(dataset.feature_manifest),
            "dataset_hash": dataset.dataset_hash,
            "horizon_hours": horizon,
            "target_definition": {
                "window": f"(T, T+{horizon}h]",
                "minimum_request_threshold_mw": cfg.minimum_request_threshold_mw,
                "failure_delivery_ratio_threshold": cfg.failure_delivery_ratio_threshold,
                "end_of_data_censoring": True,
            },
            "random_seed": cfg.random_seed,
            "classification_threshold": cfg.classification_threshold,
            "selection_policy": "validation PR-AUC, then Brier, fixed simplicity order",
            "calibration_method": calibration_method,
            "calibration_fit_partition": "chronological held-out calibration only",
            "validation_comparison": comparison,
            "calibration_comparison": calibration_comparison,
            "test_metrics": test_metrics,
            "training_prevalence": float(y_train.mean()),
            "training_positive_rows": int(y_train.sum()),
            "training_negative_rows": int((y_train == 0).sum()),
            "training_distinct_failure_events": training_events,
            "training_date_range": [
                str(split.train.keys.timestamp_utc.min()),
                str(split.train.keys.timestamp_utc.max()),
            ],
            "validation_date_range": [
                str(split.validation.keys.timestamp_utc.min()),
                str(split.validation.keys.timestamp_utc.max()),
            ],
            "calibration_date_range": [
                str(split.calibration.keys.timestamp_utc.min()),
                str(split.calibration.keys.timestamp_utc.max()),
            ],
            "test_date_range": [
                str(split.test.keys.timestamp_utc.min()),
                str(split.test.keys.timestamp_utc.max()),
            ],
            "hyperparameters": model_hyperparameters(base),
            "prediction_provenance": "MODEL_PREDICTION",
        },
    )
    return DeliveryRiskTrainingResult(
        artifact, split, comparison, calibration_comparison, test_metrics
    )
