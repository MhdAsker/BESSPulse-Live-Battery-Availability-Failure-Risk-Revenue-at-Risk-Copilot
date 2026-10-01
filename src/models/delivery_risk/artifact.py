"""Delivery-risk estimator wrapper and trusted-local artifact aliases."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from models.artifact import ModelArtifact, load_model_artifact, save_model_artifact
from models.delivery_risk.calibrate import ProbabilityCalibrator


@dataclass
class CalibratedRiskEstimator:
    base_estimator: Any
    calibrator: ProbabilityCalibrator

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        raw = np.asarray(self.base_estimator.predict_proba(X)[:, 1], dtype=float)
        positive = self.calibrator.predict(raw)
        return np.column_stack([1.0 - positive, positive])


__all__ = [
    "CalibratedRiskEstimator",
    "ModelArtifact",
    "load_model_artifact",
    "save_model_artifact",
]
