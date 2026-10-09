"""Deterministic drift, quality, maturity, and evaluation metrics."""

from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def freshness_seconds(latest: datetime | None, now: datetime | None = None) -> float | None:
    if latest is None:
        return None
    reference = now or datetime.now(UTC)
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return max(0.0, (reference - latest).total_seconds())


def missingness(values: FloatArray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.isnan(array).mean()) if array.size else 1.0


def population_stability_index(
    reference: FloatArray, current: FloatArray, bins: int = 10
) -> float | None:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(ref[0], cur) else None
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(ref, edges)[0] / len(ref)
    cur_hist = np.histogram(cur, edges)[0] / len(cur)
    epsilon = 1e-6
    ratio = (cur_hist + epsilon) / (ref_hist + epsilon)
    return float(np.sum((cur_hist - ref_hist) * np.log(ratio)))


def wasserstein_distance(reference: FloatArray, current: FloatArray) -> float | None:
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return None
    quantiles = np.linspace(0, 1, max(len(ref), len(cur)))
    return float(np.mean(np.abs(np.quantile(ref, quantiles) - np.quantile(cur, quantiles))))


def matured_labels(
    prediction_times: list[datetime],
    labels: list[float | None],
    horizon_hours: int,
    as_of: datetime,
) -> tuple[IntArray, FloatArray]:
    indices = [
        index
        for index, (timestamp, label) in enumerate(zip(prediction_times, labels, strict=True))
        if label is not None and timestamp + timedelta(hours=horizon_hours) <= as_of
    ]
    matured = np.asarray([labels[index] for index in indices], dtype=float)
    return np.asarray(indices, dtype=int), matured


def brier_score(probabilities: FloatArray, labels: FloatArray) -> float | None:
    if not len(labels) or len(probabilities) != len(labels):
        return None
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def rolling_brier_24h(
    probabilities: FloatArray,
    labels: FloatArray,
    prediction_times: list[datetime],
    as_of: datetime,
) -> float | None:
    start = as_of - timedelta(hours=24)
    selected = [index for index, time in enumerate(prediction_times) if start <= time <= as_of]
    if not selected:
        return None
    return brier_score(probabilities[selected], labels[selected])
