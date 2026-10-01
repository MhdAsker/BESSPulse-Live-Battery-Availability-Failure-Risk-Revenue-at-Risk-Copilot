"""Trusted-local anomaly artifact persistence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class AnomalyArtifact:
    detector_name: str
    detector_version: str
    fitted_object: Any
    feature_names: tuple[str, ...]
    threshold: float
    persistence_intervals: int
    metadata: dict[str, Any]


def save_anomaly_artifact(artifact: AnomalyArtifact, directory: str | Path) -> tuple[Path, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "detector.joblib"
    metadata_path = path / "metadata.json"
    joblib.dump(artifact, model_path)
    metadata = {key: value for key, value in asdict(artifact).items() if key != "fitted_object"}
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def load_anomaly_artifact(path: str | Path) -> AnomalyArtifact:
    artifact: Any = joblib.load(Path(path))
    if not isinstance(artifact, AnomalyArtifact):
        raise TypeError("File is not a BESSPulse anomaly artifact")
    return artifact
