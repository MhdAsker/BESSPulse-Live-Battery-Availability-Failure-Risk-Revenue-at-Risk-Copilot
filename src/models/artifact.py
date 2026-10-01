"""Trusted-local joblib artifacts with explicit feature and metadata contracts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelArtifact:
    estimator: Any
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    model_name: str
    model_version: str
    target_name: str
    metadata: dict[str, Any]


def save_model_artifact(artifact: ModelArtifact, directory: str | Path) -> tuple[Path, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "model.joblib"
    metadata_path = path / "metadata.json"
    joblib.dump(artifact, model_path)
    metadata = {
        **artifact.metadata,
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "target_name": artifact.target_name,
        "feature_names": list(artifact.feature_names),
        "numeric_features": list(artifact.numeric_features),
        "categorical_features": list(artifact.categorical_features),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def load_model_artifact(path: str | Path) -> ModelArtifact:
    """Load only trusted local project artifacts; joblib is unsafe for untrusted files."""

    artifact: Any = joblib.load(Path(path))
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("File does not contain a BESSPulse ModelArtifact")
    return artifact
