"""Predictive-association explanations for fitted delivery-risk pipelines."""

from typing import Any

import numpy as np
import pandas as pd

from models.artifact import ModelArtifact


def global_feature_importance(artifact: ModelArtifact) -> pd.DataFrame:
    base = artifact.estimator.base_estimator
    if not hasattr(base, "named_steps"):
        raise ValueError("Selected baseline does not expose model-native importance")
    preprocess = base.named_steps["preprocess"]
    classifier = base.named_steps["classifier"]
    names = preprocess.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_[0], dtype=float)
        kind = "standardized_coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        kind = "model_native_importance"
    else:
        raise ValueError("Selected classifier does not expose native importance")
    return pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "absolute_importance": np.abs(values),
            "importance_kind": kind,
            "provenance": "DERIVED MODEL EXPLANATION",
        }
    ).sort_values("absolute_importance", ascending=False, kind="mergesort")


def local_tree_shap(
    artifact: ModelArtifact, row: pd.DataFrame, *, top_n: int = 10
) -> list[dict[str, Any]]:
    """Return top SHAP associations for one row; values are not causal effects."""

    base = artifact.estimator.base_estimator
    if not hasattr(base, "named_steps"):
        raise ValueError("SHAP is available only for fitted tree pipelines")
    preprocess = base.named_steps["preprocess"]
    classifier = base.named_steps["classifier"]
    if artifact.model_name not in {"random_forest", "lightgbm", "xgboost"}:
        raise ValueError("Local SHAP support is limited to selected tree models")
    import shap

    transformed = preprocess.transform(row[list(artifact.feature_names)])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    feature_names = preprocess.get_feature_names_out()
    pairs = sorted(
        zip(feature_names, array[0], strict=True),
        key=lambda pair: abs(float(pair[1])),
        reverse=True,
    )[:top_n]
    return [
        {
            "feature": str(name),
            "shap_value": float(value),
            "provenance": "DERIVED MODEL EXPLANATION",
        }
        for name, value in pairs
    ]


def global_tree_shap(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    *,
    sample_size: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Mean absolute SHAP association on a deterministic bounded sample."""

    if artifact.model_name not in {"random_forest", "lightgbm", "xgboost"}:
        raise ValueError("Global SHAP support is limited to selected tree models")
    base = artifact.estimator.base_estimator
    preprocess = base.named_steps["preprocess"]
    classifier = base.named_steps["classifier"]
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_seed)
    transformed = preprocess.transform(sample[list(artifact.feature_names)])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    import shap

    values = shap.TreeExplainer(classifier).shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    importance = np.mean(np.abs(array), axis=0)
    return pd.DataFrame(
        {
            "feature": preprocess.get_feature_names_out(),
            "mean_absolute_shap": importance,
            "sample_size": len(sample),
            "sampling_seed": random_seed,
            "provenance": "DERIVED MODEL EXPLANATION",
        }
    ).sort_values("mean_absolute_shap", ascending=False, kind="mergesort")
