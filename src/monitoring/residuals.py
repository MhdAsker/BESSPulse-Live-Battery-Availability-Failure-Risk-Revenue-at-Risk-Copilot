"""Residual behavior summaries; drift is not classified as an asset fault."""

import numpy as np
from numpy.typing import NDArray

from monitoring.drift import numeric_drift


def residual_summary(
    reference: NDArray[np.float64], current: NDArray[np.float64]
) -> dict[str, float | None]:
    finite = np.asarray(current, dtype=float)
    finite = finite[np.isfinite(finite)]
    result = numeric_drift(reference, current)
    result.update(
        bias=float(np.mean(finite)) if len(finite) else None,
        mean_absolute=float(np.mean(np.abs(finite))) if len(finite) else None,
        p95_absolute=float(np.quantile(np.abs(finite), 0.95)) if len(finite) else None,
    )
    return result
