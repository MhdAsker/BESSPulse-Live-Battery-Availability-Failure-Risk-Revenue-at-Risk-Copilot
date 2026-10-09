"""Post-training MLflow logging command for completed, validated experiments."""

import argparse
import json
from pathlib import Path
from typing import Any

from tracking.service import ExperimentTracker

REQUIRED_LINEAGE = {
    "dataset_hash",
    "feature_set_version",
    "target_definition_version",
    "artifact_version",
}


def _load_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def log_completed_experiment(
    family: str,
    metadata_path: Path,
    metrics_path: Path,
    artifacts: tuple[Path, ...] = (),
    tracker: ExperimentTracker | None = None,
) -> str | None:
    """Log existing outputs; never rerun training or invent absent metrics."""

    metadata = _load_object(metadata_path)
    missing = sorted(REQUIRED_LINEAGE - metadata.keys())
    if missing:
        raise ValueError(f"Missing required lineage fields: {', '.join(missing)}")
    metrics = {
        key: float(value)
        for key, value in _load_object(metrics_path).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    client = tracker or ExperimentTracker()
    with client.run(family, parameters=metadata, tags={"run_kind": "completed_training"}) as run:
        if run is None:
            return None
        client.log_metrics(metrics)
        client.log_artifact(metadata_path, "metadata")
        client.log_artifact(metrics_path, "metrics")
        for artifact in artifacts:
            client.log_artifact(artifact)
        return str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a completed BESSPulse experiment to MLflow")
    parser.add_argument("--family", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()
    run_id = log_completed_experiment(
        args.family,
        args.metadata,
        args.metrics,
        tuple(args.artifact),
    )
    print(json.dumps({"status": "logged" if run_id else "tracking_unavailable", "run_id": run_id}))


if __name__ == "__main__":
    main()
