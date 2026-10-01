"""Leave-one-out robust peer anomaly scoring with explicit zero-MAD behavior."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from models.anomaly.common import interval_output
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import AnomalyDataset

ROBUST_COLUMNS = (
    "temperature_peer_robust_zscore",
    "voltage_spread_peer_robust_zscore",
    "rte_peer_robust_zscore",
    "soc_peer_robust_zscore",
    "power_tracking_peer_robust_zscore",
)


def robust_zscore(value: float, peers: NDArray[np.float64], epsilon: float) -> float:
    valid = peers[np.isfinite(peers)]
    if not np.isfinite(value) or not len(valid):
        return float("nan")
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    deviation = value - median
    if mad <= epsilon:
        return 0.0 if abs(deviation) <= epsilon else float("nan")
    return deviation / (1.4826 * mad)


def score_robust_peer(dataset: AnomalyDataset, config: AnomalyConfig | None = None) -> pd.DataFrame:
    cfg = config or AnomalyConfig()
    columns = [column for column in ROBUST_COLUMNS if column in dataset.predictors]
    if not columns:
        raise ValueError("Robust detector requires canonical peer z-score features")
    absolute = dataset.predictors[columns].abs().replace([np.inf, -np.inf], np.nan)
    if "peer_count" in dataset.predictors:
        absolute = absolute.where(dataset.predictors["peer_count"].ge(cfg.minimum_peer_count))
    score = absolute.max(axis=1, skipna=True)
    score = score.where(absolute.notna().any(axis=1))
    supporting = [
        [
            column
            for column in columns
            if pd.notna(absolute.at[index, column])
            and absolute.at[index, column] == score.iloc[index]
        ]
        for index in range(len(score))
    ]
    output = interval_output(
        dataset.keys,
        score,
        detector_name="robust_peer",
        detector_version="robust_peer_v1",
        supporting_signals=supporting,
    )
    output["zero_mad_policy"] = "equal->0; deviating->NaN"
    output["minimum_peer_count"] = cfg.minimum_peer_count
    return output
