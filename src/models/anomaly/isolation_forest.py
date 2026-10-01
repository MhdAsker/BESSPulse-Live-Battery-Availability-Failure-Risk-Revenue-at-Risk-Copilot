"""Healthy-cohort Isolation Forest with high-is-anomalous project scores."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from models.anomaly.common import interval_output
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import AnomalyDataset, healthy_mask

IFOREST_FEATURES = (
    "temperature_vs_ambient_c",
    "temperature_ramp_c",
    "temperature_spread_c",
    "voltage_spread_v",
    "rack_power_residual_mw",
    "rte",
    "rte_trend",
    "temperature_peer_deviation",
    "soc_peer_deviation",
    "power_tracking_peer_deviation",
    "thermal_residual_c",
    "delivery_shortfall_mw",
    "alarm_frequency",
    "availability_rate",
)


@dataclass
class IsolationForestArtifact:
    estimator: Any
    feature_names: tuple[str, ...]
    version: str
    dataset_hash: str
    training_start: str
    training_end: str
    random_seed: int


def fit_isolation_forest(
    train: AnomalyDataset, config: AnomalyConfig | None = None
) -> IsolationForestArtifact:
    cfg = config or AnomalyConfig()
    features = tuple(column for column in IFOREST_FEATURES if column in train.predictors)
    healthy = healthy_mask(train)
    X = train.predictors.loc[healthy, list(features)]
    if len(X) < cfg.minimum_training_rows:
        raise ValueError("Insufficient healthy Isolation Forest training rows")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=cfg.iforest_n_estimators,
                    max_samples=min(cfg.iforest_max_samples, len(X)),
                    contamination=cfg.iforest_contamination,
                    random_state=cfg.random_seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X)
    return IsolationForestArtifact(
        pipeline,
        features,
        "isolation_forest_v1",
        train.dataset_hash,
        str(train.keys.timestamp_utc.min()),
        str(train.keys.timestamp_utc.max()),
        cfg.random_seed,
    )


def score_isolation_forest(
    dataset: AnomalyDataset, artifact: IsolationForestArtifact
) -> pd.DataFrame:
    X = dataset.predictors[list(artifact.feature_names)]
    score = -artifact.estimator.decision_function(X)
    return interval_output(
        dataset.keys,
        pd.Series(score, index=dataset.keys.index),
        detector_name="isolation_forest",
        detector_version=artifact.version,
        supporting_signals=[list(artifact.feature_names) for _ in range(len(X))],
        provenance="MODEL_PREDICTION / ANOMALY MODEL OUTPUT",
    )
