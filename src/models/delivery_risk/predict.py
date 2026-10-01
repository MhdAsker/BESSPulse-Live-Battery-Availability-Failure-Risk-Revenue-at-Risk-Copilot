"""Validated single- and multi-horizon delivery-risk inference."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from features.validation import is_prohibited_column
from models.artifact import ModelArtifact


def _validated_features(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    leakage = [
        column
        for column in frame
        if is_prohibited_column(str(column))
        or str(column).lower().startswith("future_")
        or str(column).lower() == "time_to_next_failure"
    ]
    if leakage:
        raise ValueError(f"Inference contains leakage fields: {leakage}")
    missing = [column for column in artifact.feature_names if column not in frame]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    features = frame[list(artifact.feature_names)].copy()
    for column in artifact.numeric_features:
        converted = pd.to_numeric(features[column], errors="coerce")
        if (features[column].notna() & converted.isna()).any():
            raise ValueError(f"Incompatible numeric feature: {column}")
        features[column] = converted
    numeric = features[list(artifact.numeric_features)].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Inference features contain infinite values")
    return features


def predict_horizon(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["timestamp_utc", "asset_id"]:
        if column not in frame:
            raise ValueError(f"Missing prediction key: {column}")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    horizon = int(artifact.metadata["horizon_hours"])
    probability = artifact.estimator.predict_proba(_validated_features(artifact, frame))[:, 1]
    result = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset_id": frame["asset_id"].astype(str),
            f"failure_probability_{horizon}h": probability,
            f"model_version_{horizon}h": artifact.model_version,
        }
    )
    result["feature_set_version"] = artifact.metadata["feature_set_version"]
    result["data_provenance"] = "MODEL_PREDICTION"
    return result


def predict_delivery_risk(
    artifacts: Mapping[int, ModelArtifact], frame: pd.DataFrame
) -> pd.DataFrame:
    """Return distinct probabilities; a missing horizon is never substituted."""

    if not artifacts:
        raise ValueError("At least one horizon artifact is required")
    expected = {6, 12, 24}
    missing = sorted(expected.difference(artifacts))
    if missing:
        raise ValueError(f"Missing delivery-risk models for horizons: {missing}")
    output: pd.DataFrame | None = None
    for horizon in sorted(artifacts):
        prediction = predict_horizon(artifacts[horizon], frame)
        common = ["timestamp_utc", "asset_id", "feature_set_version", "data_provenance"]
        if output is None:
            output = prediction
        else:
            output = output.merge(prediction, on=common, validate="one_to_one")
    if output is None:  # pragma: no cover - protected above
        raise AssertionError("No prediction frame produced")
    return output
