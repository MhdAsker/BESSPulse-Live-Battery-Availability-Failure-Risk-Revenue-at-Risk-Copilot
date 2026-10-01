"""Independent LightGBM quantile models and explicit crossing handling."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from models.price.config import PriceModelConfig
from models.price.dataset import PriceDataset, PriceSplit, chronological_price_split
from models.price.evaluate import quantile_metrics
from models.price.train import lightgbm_regressor


@dataclass(frozen=True)
class QuantileResult:
    estimators: dict[float, object]
    raw_predictions: dict[float, NDArray[np.float64]]
    monotonic_predictions: dict[float, NDArray[np.float64]]
    metrics: dict[str, object]


def train_quantile_models(
    dataset: PriceDataset,
    config: PriceModelConfig | None = None,
    split: PriceSplit | None = None,
) -> QuantileResult:
    cfg = config or PriceModelConfig()
    partitions = split or chronological_price_split(dataset, cfg)
    estimators: dict[float, object] = {}
    raw: dict[float, NDArray[np.float64]] = {}
    for level in cfg.quantile_levels:
        estimator = lightgbm_regressor(cfg, objective="quantile", alpha=level)
        estimator.fit(partitions.train.predictors, partitions.train.target)
        estimators[level] = estimator
        raw[level] = np.asarray(estimator.predict(partitions.test.predictors), dtype=float)
    matrix = np.sort(np.column_stack([raw[level] for level in cfg.quantile_levels]), axis=1)
    monotonic = {level: matrix[:, index] for index, level in enumerate(cfg.quantile_levels)}
    metrics = quantile_metrics(partitions.test.target, raw)
    metrics["postprocessing"] = "row-wise sort; raw predictions retained for metrics"
    return QuantileResult(estimators, raw, monotonic, metrics)
