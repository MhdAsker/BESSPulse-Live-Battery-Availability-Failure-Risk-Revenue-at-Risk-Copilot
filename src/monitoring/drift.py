"""Numeric reference-versus-current drift summaries."""

import numpy as np
from numpy.typing import NDArray

from monitoring.metrics import population_stability_index, wasserstein_distance


def numeric_drift(
    reference: NDArray[np.float64], current: NDArray[np.float64]
) -> dict[str, float | None]:
    return {
        "psi": population_stability_index(reference, current),
        "wasserstein": wasserstein_distance(reference, current),
    }
