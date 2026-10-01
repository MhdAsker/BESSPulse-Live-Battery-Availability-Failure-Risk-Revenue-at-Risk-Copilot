"""Batch expected-power inference and causal residual features."""

import numpy as np
import pandas as pd

from features.validation import trailing_statistic
from models.artifact import ModelArtifact


def _validated_predictors(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in artifact.feature_names if column not in frame]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    X = frame[list(artifact.feature_names)].copy()
    for column in artifact.numeric_features:
        converted = pd.to_numeric(X[column], errors="coerce")
        if (X[column].notna() & converted.isna()).any():
            raise ValueError(f"Incompatible numeric feature: {column}")
        X[column] = converted
    values = X[list(artifact.numeric_features)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Inference features contain infinite values")
    return X


def predict_expected_power(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict without requiring actual_power_mw."""

    for key in ("timestamp_utc", "asset_id"):
        if key not in frame:
            raise ValueError(f"Missing prediction key: {key}")
    X = _validated_predictors(artifact, frame)
    result = frame[["timestamp_utc", "asset_id"]].copy()
    result["expected_actual_power_mw"] = artifact.estimator.predict(X)
    result["model_version"] = artifact.model_version
    result["model_provenance"] = "MODEL_PREDICTION"
    return result


def add_power_residuals(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    rated_power_mw: float = 20.0,
    window: str = "1h",
    persistence_threshold_mw: float = 0.5,
) -> pd.DataFrame:
    required = ["timestamp_utc", "asset_id", "actual_power_mw", "requested_power_mw"]
    missing = [column for column in required if column not in actual]
    if missing:
        raise ValueError(f"Missing residual columns: {missing}")
    result = predictions.merge(
        actual[required], on=["timestamp_utc", "asset_id"], validate="one_to_one"
    )
    result["power_residual_mw"] = result["actual_power_mw"] - result["expected_actual_power_mw"]
    result["absolute_power_residual_mw"] = result["power_residual_mw"].abs()
    result["normalized_power_residual"] = result["power_residual_mw"] / rated_power_mw
    directional_shortfall = np.sign(result["requested_power_mw"]) * (
        result["expected_actual_power_mw"] - result["actual_power_mw"]
    )
    result["delivery_shortfall_mw"] = directional_shortfall.clip(lower=0)
    result = result.sort_values(["asset_id", "timestamp_utc"], kind="mergesort").reset_index(
        drop=True
    )
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("asset_id", sort=False):
        timestamps = group["timestamp_utc"]
        group["power_residual_rolling_mean"] = trailing_statistic(
            group["power_residual_mw"],
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["power_residual_rolling_std"] = trailing_statistic(
            group["power_residual_mw"],
            timestamps,
            window,
            "std",
            min_periods=2,
            require_full_window=False,
        )
        group["negative_residual_fraction"] = trailing_statistic(
            (group["power_residual_mw"] < 0).astype(float),
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["delivery_shortfall_rolling_mean"] = trailing_statistic(
            group["delivery_shortfall_mw"],
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["residual_persistence_count"] = trailing_statistic(
            (group["delivery_shortfall_mw"] > persistence_threshold_mw).astype(float),
            timestamps,
            window,
            "sum",
            min_periods=1,
            require_full_window=False,
        )
        pieces.append(group)
    output = pd.concat(pieces, ignore_index=True)
    output["prediction_provenance"] = "MODEL_PREDICTION"
    output["residual_provenance"] = "DERIVED"
    return output
