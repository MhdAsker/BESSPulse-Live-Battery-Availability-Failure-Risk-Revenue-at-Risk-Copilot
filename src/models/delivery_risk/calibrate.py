"""Chronologically held-out probability calibration."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Any | None = None

    def predict(self, probability: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(probability, dtype=float)
        if self.method == "identity":
            calibrated = values
        elif self.method == "platt":
            if self.model is None:
                raise ValueError("Platt calibrator has no fitted model")
            calibrated = self.model.predict_proba(values.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            if self.model is None:
                raise ValueError("Isotonic calibrator has no fitted model")
            calibrated = self.model.predict(values)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def fit_calibrators(
    probability: NDArray[np.float64], actual: NDArray[Any], *, random_seed: int
) -> dict[str, ProbabilityCalibrator]:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(actual, dtype=int)
    if len(np.unique(y)) < 2:
        return {"identity": ProbabilityCalibrator("identity")}
    platt = LogisticRegression(random_state=random_seed).fit(p.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return {
        "identity": ProbabilityCalibrator("identity"),
        "platt": ProbabilityCalibrator("platt", platt),
        "isotonic": ProbabilityCalibrator("isotonic", isotonic),
    }
