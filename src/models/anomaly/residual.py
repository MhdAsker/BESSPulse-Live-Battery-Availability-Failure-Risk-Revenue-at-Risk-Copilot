"""Train-only robust expected-behavior residual reference detector."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.anomaly.common import interval_output
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import AnomalyDataset, healthy_mask

RESIDUAL_FEATURES = (
    "thermal_residual_c",
    "thermal_residual_rolling_mean",
    "delivery_shortfall_mw",
)


def directional_delivery_shortfall(
    requested_power_mw: pd.Series, actual_power_mw: pd.Series
) -> pd.Series:
    """Return positive under-delivery for either charging or discharging requests."""
    direction = np.sign(requested_power_mw)
    shortfall: pd.Series = (direction * (requested_power_mw - actual_power_mw)).clip(lower=0)
    return shortfall


@dataclass(frozen=True)
class ResidualReference:
    centers: dict[str, float]
    scales: dict[str, float]
    feature_names: tuple[str, ...]
    training_start: str
    training_end: str
    dataset_hash: str
    version: str = "residual_v1"


def fit_residual_reference(
    train: AnomalyDataset, config: AnomalyConfig | None = None
) -> ResidualReference:
    cfg = config or AnomalyConfig()
    features = tuple(str(column) for column in RESIDUAL_FEATURES if column in train.predictors)
    if not features:
        raise ValueError("Residual detector requires leakage-safe residual features")
    provenance_ok = "expected_temperature_c" in train.predictors
    if not provenance_ok:
        raise ValueError("Prompt 4 expected prediction is required for residual provenance")
    healthy = healthy_mask(train)
    if int(healthy.sum()) < cfg.minimum_training_rows:
        raise ValueError("Insufficient healthy rows for residual reference")
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in features:
        values = train.predictors.loc[healthy, column].dropna().to_numpy(dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        centers[column] = center
        scales[column] = max(1.4826 * mad, cfg.mad_epsilon)
    return ResidualReference(
        centers,
        scales,
        features,
        str(train.keys.timestamp_utc.min()),
        str(train.keys.timestamp_utc.max()),
        train.dataset_hash,
    )


def score_residual(dataset: AnomalyDataset, reference: ResidualReference) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for column in reference.feature_names:
        values = dataset.predictors[column]
        z = (values - reference.centers[column]) / reference.scales[column]
        scores[column] = z.clip(lower=0) if column != "delivery_shortfall_mw" else z.abs()
    matrix = pd.DataFrame(scores)
    score = matrix.max(axis=1, skipna=True).where(matrix.notna().any(axis=1))
    supporting = [
        [
            str(column)
            for column in matrix
            if pd.notna(matrix.at[index, column]) and matrix.at[index, column] == score.iloc[index]
        ]
        for index in range(len(score))
    ]
    return interval_output(
        dataset.keys,
        score,
        detector_name="residual",
        detector_version=reference.version,
        supporting_signals=supporting,
    )
