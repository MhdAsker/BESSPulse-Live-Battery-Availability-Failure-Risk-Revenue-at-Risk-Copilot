"""Observable, leave-one-out rack peer analytics."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from besspulse.config import SimulationConfig
from features.validation import (
    is_prohibited_column,
    prepare_temporal_frame,
    validate_feature_frame,
)

_PEER_METRICS: dict[str, str] = {
    "temperature": "temperature_mean_c",
    "voltage_spread": "voltage_spread_v",
    "rte": "rte",
    "soc": "soc",
    "power_tracking": "rack_power_residual_mw",
}


def _percentile(value: float, peers: NDArray[np.float64]) -> float:
    """Midrank empirical percentile: (less + 0.5 * equal) / peer_count."""

    less = float(np.sum(peers < value))
    equal = float(np.sum(peers == value))
    return (less + 0.5 * equal) / len(peers)


def _peer_statistics(
    value: float, peers: NDArray[np.float64], epsilon: float
) -> tuple[float, float, float, float]:
    median = float(np.median(peers))
    deviation = value - median
    mad = float(np.median(np.abs(peers - median)))
    if mad <= epsilon:
        robust_z = 0.0 if abs(deviation) <= epsilon else float("nan")
    else:
        robust_z = deviation / (1.4826 * mad)
    return median, deviation, robust_z, _percentile(value, peers)


def add_peer_features(
    rack_features: pd.DataFrame,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Add same-PCS peer baselines with observable site fallback and no self inclusion."""

    cfg = config or SimulationConfig()
    prohibited = [
        column
        for column in rack_features.columns
        if isinstance(column, str) and is_prohibited_column(column)
    ]
    clean = rack_features.drop(columns=prohibited)
    required = ["timestamp_utc", "rack_id", "pcs_id", "availability", *_PEER_METRICS.values()]
    missing = [column for column in required if column not in clean]
    if missing:
        raise ValueError(f"Missing peer columns: {missing}")
    frame = prepare_temporal_frame(clean, ["rack_id"])
    for prefix in _PEER_METRICS:
        frame[f"{prefix}_peer_median"] = np.nan
        frame[f"{prefix}_peer_deviation"] = np.nan
        frame[f"{prefix}_peer_robust_zscore"] = np.nan
        frame[f"{prefix}_peer_percentile"] = np.nan
    frame["peer_count"] = 0
    frame["peer_valid_fraction"] = np.nan
    frame["peer_group"] = pd.Series(pd.NA, index=frame.index, dtype="string")

    minimum = cfg.features.peer_min_population
    epsilon = cfg.features.peer_epsilon
    for _, timestamp_rows in frame.groupby("timestamp_utc", sort=False):
        for target_index, target in timestamp_rows.iterrows():
            same_pcs = timestamp_rows.loc[
                (timestamp_rows["pcs_id"] == target["pcs_id"])
                & (timestamp_rows["rack_id"] != target["rack_id"])
            ]
            pcs_eligible = same_pcs.loc[same_pcs["availability"].astype(bool)]
            if len(pcs_eligible) >= minimum:
                candidates = same_pcs
                eligible = pcs_eligible
                group_name: str | None = "PCS"
            else:
                candidates = timestamp_rows.loc[timestamp_rows["rack_id"] != target["rack_id"]]
                eligible = candidates.loc[candidates["availability"].astype(bool)]
                group_name = "SITE" if len(eligible) >= minimum else None
            frame.at[target_index, "peer_count"] = len(eligible)
            frame.at[target_index, "peer_valid_fraction"] = (
                len(eligible) / len(candidates) if len(candidates) else np.nan
            )
            if group_name is None:
                continue
            frame.at[target_index, "peer_group"] = group_name
            for prefix, source in _PEER_METRICS.items():
                peer_values = eligible[source].dropna().to_numpy(dtype=float)
                target_value = target[source]
                if len(peer_values) < minimum or pd.isna(target_value):
                    continue
                median, deviation, robust_z, percentile = _peer_statistics(
                    float(target_value), peer_values, epsilon
                )
                frame.at[target_index, f"{prefix}_peer_median"] = median
                frame.at[target_index, f"{prefix}_peer_deviation"] = deviation
                frame.at[target_index, f"{prefix}_peer_robust_zscore"] = robust_z
                frame.at[target_index, f"{prefix}_peer_percentile"] = percentile
    result = frame.sort_values(["timestamp_utc", "rack_id"], kind="mergesort").reset_index(
        drop=True
    )
    result.attrs = {
        "data_provenance": "DERIVED",
        "entity_level": "rack",
        "peer_strategy": "leave_one_out_same_pcs_then_site_fallback",
    }
    validate_feature_frame(result, ["rack_id"])
    return result
