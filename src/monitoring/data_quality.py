"""Transparent numeric data-quality summaries."""

import numpy as np
from numpy.typing import NDArray


def quality_fractions(values: NDArray[np.float64]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"missing_fraction": 1.0, "infinite_fraction": 0.0, "invalid_fraction": 1.0}
    missing = np.isnan(array)
    infinite = np.isinf(array)
    invalid = missing | infinite
    return {
        "missing_fraction": float(missing.mean()),
        "infinite_fraction": float(infinite.mean()),
        "invalid_fraction": float(invalid.mean()),
    }
