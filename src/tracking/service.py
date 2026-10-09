"""Secret-safe MLflow tracking with a local default and graceful remote failure."""

import importlib
import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SENSITIVE = re.compile(r"(secret|token|password|credential|api.?key|database.?url|auth)", re.I)
EXPERIMENTS = {
    "expected_power": "BESSPulse-ExpectedPower",
    "expected_temperature": "BESSPulse-ExpectedTemperature",
    "anomaly": "BESSPulse-Anomaly",
    "delivery_risk": "BESSPulse-DeliveryRisk",
    "delivery_risk_6h": "BESSPulse-DeliveryRisk-6h",
    "delivery_risk_12h": "BESSPulse-DeliveryRisk-12h",
    "delivery_risk_24h": "BESSPulse-DeliveryRisk-24h",
    "price": "BESSPulse-PriceForecast",
}


def _mlflow() -> Any:
    """Load the optional SDK without coupling analysis to its dependency tree."""

    return importlib.import_module("mlflow")


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    enabled: bool = True
    fail_open: bool = True


def sanitize_mapping(values: Mapping[str, Any]) -> dict[str, str]:
    """Flatten safe scalar metadata and exclude secret-bearing keys."""

    return {
        str(key): str(value)[:500]
        for key, value in values.items()
        if not SENSITIVE.search(str(key)) and isinstance(value, str | int | float | bool)
    }


class ExperimentTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    @contextmanager
    def run(
        self,
        family: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> Iterator[Any | None]:
        if not self.config.enabled:
            yield None
            return
        run_context: Any | None = None
        try:
            mlflow = _mlflow()
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(EXPERIMENTS.get(family, f"besspulse-{family}"))
            run_context = mlflow.start_run()
            active = run_context.__enter__()
            if parameters:
                mlflow.log_params(sanitize_mapping(parameters))
            safe_tags = sanitize_mapping(tags or {})
            if family == "delivery_risk_24h":
                safe_tags["calibration_status"] = "limited"
            mlflow.set_tags(safe_tags)
        except Exception as exc:
            if run_context is not None:
                run_context.__exit__(type(exc), exc, exc.__traceback__)
            if not self.config.fail_open:
                raise
            yield None
            return
        try:
            yield active
        except BaseException:
            run_context.__exit__(*sys.exc_info())
            raise
        else:
            run_context.__exit__(None, None, None)

    @staticmethod
    def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
        mlflow = _mlflow()
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    @staticmethod
    def log_artifact(path: Path, artifact_path: str = "artifacts") -> None:
        mlflow = _mlflow()
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    @staticmethod
    def register_candidate(model_uri: str, name: str, validated: bool = False) -> Any:
        """Register only explicitly validated candidates; never auto-promote."""

        if not validated:
            raise ValueError("Model registration requires explicit validation.")
        mlflow = _mlflow()
        return mlflow.register_model(model_uri, name)
