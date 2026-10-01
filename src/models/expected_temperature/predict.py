"""Batch expected-temperature inference and causal residual features."""

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
    if np.isinf(X[list(artifact.numeric_features)].to_numpy(dtype=float)).any():
        raise ValueError("Inference features contain infinite values")
    return X


def predict_expected_temperature(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict without requiring temperature_mean_c or a known rack identity feature."""

    for key in ("timestamp_utc", "rack_id", "pcs_id"):
        if key not in frame:
            raise ValueError(f"Missing prediction key: {key}")
    X = _validated_predictors(artifact, frame)
    result = frame[["timestamp_utc", "rack_id", "pcs_id"]].copy()
    result["expected_temperature_c"] = artifact.estimator.predict(X)
    result["model_version"] = artifact.model_version
    result["model_provenance"] = "MODEL_PREDICTION"
    return result


def add_thermal_residuals(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    window: str = "1h",
    persistence_threshold_c: float = 1.0,
) -> pd.DataFrame:
    required = ["timestamp_utc", "rack_id", "temperature_mean_c"]
    missing = [column for column in required if column not in actual]
    if missing:
        raise ValueError(f"Missing residual columns: {missing}")
    result = predictions.merge(
        actual[required], on=["timestamp_utc", "rack_id"], validate="one_to_one"
    )
    result["thermal_residual_c"] = result["temperature_mean_c"] - result["expected_temperature_c"]
    result["absolute_thermal_residual_c"] = result["thermal_residual_c"].abs()
    result = result.sort_values(["rack_id", "timestamp_utc"], kind="mergesort").reset_index(
        drop=True
    )
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby("rack_id", sort=False):
        timestamps = group["timestamp_utc"]
        group["thermal_residual_rolling_mean"] = trailing_statistic(
            group["thermal_residual_c"],
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["thermal_residual_rolling_std"] = trailing_statistic(
            group["thermal_residual_c"],
            timestamps,
            window,
            "std",
            min_periods=2,
            require_full_window=False,
        )
        group["positive_thermal_residual_fraction"] = trailing_statistic(
            (group["thermal_residual_c"] > 0).astype(float),
            timestamps,
            window,
            "mean",
            min_periods=1,
            require_full_window=False,
        )
        group["thermal_residual_persistence_count"] = trailing_statistic(
            (group["thermal_residual_c"] > persistence_threshold_c).astype(float),
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
