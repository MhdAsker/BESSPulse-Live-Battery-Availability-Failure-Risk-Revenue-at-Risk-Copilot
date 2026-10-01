"""Shared chronological splitting, preprocessing, candidates, and evaluation."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.validation import validate_no_leakage


class ExpectedModelConfig(BaseModel):
    """Modest, reproducible candidate and temporal-split configuration."""

    model_config = ConfigDict(extra="forbid")
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)
    test_fraction: float = Field(default=0.20, gt=0, lt=1)
    selection_rmse_tolerance: float = Field(default=0.02, ge=0)
    minimum_training_rows: int = Field(default=60, ge=10)
    walk_forward_block_timestamps: int = Field(default=12, ge=1)
    random_seed: int = 42
    healthy_labels: tuple[str, ...] = ("NORMAL", "normal", "healthy")
    require_healthy_labels: bool = True
    ridge_alpha: float = Field(default=1.0, gt=0)
    rf_n_estimators: int = Field(default=80, ge=10)
    rf_max_depth: int = Field(default=8, ge=1)
    rf_min_samples_leaf: int = Field(default=2, ge=1)
    lgbm_n_estimators: int = Field(default=100, ge=10)
    lgbm_learning_rate: float = Field(default=0.05, gt=0)
    lgbm_num_leaves: int = Field(default=15, ge=2)
    residual_window: str = "1h"
    residual_persistence_threshold: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> "ExpectedModelConfig":
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("train/validation/test fractions must sum to one")
        if pd.Timedelta(self.residual_window) <= pd.Timedelta(0):
            raise ValueError("residual_window must be positive")
        return self


@dataclass(frozen=True)
class ModelDataset:
    keys: pd.DataFrame
    predictors: pd.DataFrame
    target: pd.Series
    healthy: pd.Series
    evaluation_metadata: pd.DataFrame
    target_name: str
    dataset_hash: str

    def subset(self, positions: NDArray[np.intp]) -> "ModelDataset":
        return ModelDataset(
            keys=self.keys.iloc[positions].reset_index(drop=True),
            predictors=self.predictors.iloc[positions].reset_index(drop=True),
            target=self.target.iloc[positions].reset_index(drop=True),
            healthy=self.healthy.iloc[positions].reset_index(drop=True),
            evaluation_metadata=self.evaluation_metadata.iloc[positions].reset_index(drop=True),
            target_name=self.target_name,
            dataset_hash=self.dataset_hash,
        )


@dataclass(frozen=True)
class DatasetSplit:
    train: ModelDataset
    validation: ModelDataset
    test: ModelDataset


class ColumnBaseline:
    """Naive estimator that returns one explicit predictor unchanged."""

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnBaseline":
        if self.column not in X:
            raise ValueError(f"Baseline predictor is missing: {self.column}")
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"column": self.column}

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return pd.to_numeric(X[self.column], errors="raise").to_numpy(dtype=float)


def dataset_hash(keys: pd.DataFrame, predictors: pd.DataFrame, target: pd.Series) -> str:
    combined = pd.concat([keys, predictors, target.rename("__target__")], axis=1)
    schema = "|".join(f"{name}:{combined[name].dtype}" for name in combined.columns)
    values = pd.util.hash_pandas_object(combined, index=False).to_numpy().tobytes()
    return hashlib.sha256(schema.encode() + values).hexdigest()


def chronological_split(dataset: ModelDataset, config: ExpectedModelConfig) -> DatasetSplit:
    """Split by unique timestamps; every train time precedes validation and test."""

    timestamps = pd.DatetimeIndex(dataset.keys["timestamp_utc"].sort_values().unique())
    if len(timestamps) < 5:
        raise ValueError("Too few unique timestamps for chronological three-way split")
    train_cut = max(1, int(len(timestamps) * config.train_fraction))
    validation_cut = max(
        train_cut + 1, int(len(timestamps) * (config.train_fraction + config.validation_fraction))
    )
    validation_cut = min(validation_cut, len(timestamps) - 1)

    def protect_event_boundary(cut: int) -> int:
        if "fault_id" not in dataset.evaluation_metadata or cut >= len(timestamps):
            return cut
        boundary = timestamps[cut]
        event_frame = pd.DataFrame(
            {
                "timestamp_utc": dataset.keys["timestamp_utc"],
                "fault_id": dataset.evaluation_metadata["fault_id"],
            }
        ).dropna(subset=["fault_id"])
        for _, event_rows in event_frame.groupby("fault_id"):
            event_start = event_rows["timestamp_utc"].min()
            event_end = event_rows["timestamp_utc"].max()
            if event_start < boundary <= event_end:
                return max(1, int(timestamps.searchsorted(event_start)))
        return cut

    train_cut = protect_event_boundary(train_cut)
    validation_cut = max(train_cut + 1, protect_event_boundary(validation_cut))
    validation_cut = min(validation_cut, len(timestamps) - 1)
    train_end = timestamps[train_cut - 1]
    validation_end = timestamps[validation_cut - 1]
    timestamp_values = dataset.keys["timestamp_utc"]
    train_positions = np.flatnonzero((timestamp_values <= train_end).to_numpy())
    validation_positions = np.flatnonzero(
        ((timestamp_values > train_end) & (timestamp_values <= validation_end)).to_numpy()
    )
    test_positions = np.flatnonzero((timestamp_values > validation_end).to_numpy())
    if not len(train_positions) or not len(validation_positions) or not len(test_positions):
        raise ValueError("Chronological split produced an empty partition")
    split = DatasetSplit(
        dataset.subset(train_positions),
        dataset.subset(validation_positions),
        dataset.subset(test_positions),
    )
    if not (
        split.train.keys["timestamp_utc"].max()
        < split.validation.keys["timestamp_utc"].min()
        <= split.validation.keys["timestamp_utc"].max()
        < split.test.keys["timestamp_utc"].min()
    ):
        raise AssertionError("Chronological partition ordering failed")
    return split


def healthy_training_rows(
    dataset: ModelDataset, config: ExpectedModelConfig
) -> tuple[pd.DataFrame, pd.Series]:
    mask = dataset.healthy & dataset.target.notna()
    X = dataset.predictors.loc[mask].reset_index(drop=True)
    y = dataset.target.loc[mask].reset_index(drop=True)
    if len(X) < config.minimum_training_rows:
        raise ValueError(
            f"Insufficient healthy training rows: {len(X)} < {config.minimum_training_rows}"
        )
    if y.nunique(dropna=True) < 2 or float(y.std()) < 1e-12:
        raise ValueError("Healthy training target is constant or effectively constant")
    validate_no_leakage(list(X.columns))
    return X, y


def _preprocessor(
    numeric_features: list[str], categorical_features: list[str], *, scale: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )


def build_candidate_estimators(
    numeric_features: list[str],
    categorical_features: list[str],
    baseline_column: str,
    config: ExpectedModelConfig,
) -> dict[str, Any]:
    return {
        "naive": ColumnBaseline(baseline_column),
        "ridge_linear": Pipeline(
            [
                ("preprocess", _preprocessor(numeric_features, categorical_features, scale=True)),
                # Correlated lag/power predictors made ordinary least squares numerically
                # unstable under held-out thermal faults; Ridge preserves a linear baseline.
                ("regressor", Ridge(alpha=config.ridge_alpha)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor(numeric_features, categorical_features, scale=False)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=config.rf_n_estimators,
                        max_depth=config.rf_max_depth,
                        min_samples_leaf=config.rf_min_samples_leaf,
                        random_state=config.random_seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("preprocess", _preprocessor(numeric_features, categorical_features, scale=False)),
                (
                    "regressor",
                    LGBMRegressor(
                        n_estimators=config.lgbm_n_estimators,
                        learning_rate=config.lgbm_learning_rate,
                        num_leaves=config.lgbm_num_leaves,
                        random_state=config.random_seed,
                        n_jobs=1,
                        deterministic=True,
                        force_col_wise=True,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(actual: pd.Series, prediction: NDArray[np.float64]) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    residual = actual_values - prediction
    return {
        "mae": float(mean_absolute_error(actual_values, prediction)),
        "rmse": float(mean_squared_error(actual_values, prediction) ** 0.5),
        "r2": float(r2_score(actual_values, prediction)),
        "bias_mean_residual": float(np.mean(residual)),
        "p95_absolute_residual": float(np.percentile(np.abs(residual), 95)),
    }


def select_simplest_within_tolerance(
    comparison: dict[str, dict[str, float]], tolerance: float
) -> str:
    best = min(metrics["rmse"] for metrics in comparison.values())
    eligible = {
        name for name, metrics in comparison.items() if metrics["rmse"] <= best * (1 + tolerance)
    }
    order = ["naive", "ridge_linear", "random_forest", "lightgbm"]
    return next(name for name in order if name in eligible)


def model_hyperparameters(estimator: Any) -> dict[str, Any]:
    params = estimator.get_params(deep=False)
    return cast(dict[str, Any], json.loads(json.dumps(params, default=str)))
