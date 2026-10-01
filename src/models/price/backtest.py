"""Expanding-window, rolling-origin historical price backtesting."""

from typing import Any

import numpy as np
import pandas as pd

from models.price.config import PriceModelConfig
from models.price.dataset import PriceDataset
from models.price.evaluate import price_metrics
from models.price.train import candidate_estimators


def walk_forward_backtest(
    dataset: PriceDataset,
    model_name: str,
    config: PriceModelConfig | None = None,
) -> pd.DataFrame:
    """Refit on strictly earlier rows at each origin and forecast a timestamp horizon."""

    cfg = config or PriceModelConfig()
    timestamps = pd.DatetimeIndex(dataset.timestamps)
    first_origin = timestamps.min() + pd.Timedelta(hours=cfg.minimum_training_history_hours)
    last_origin = timestamps.max() - pd.Timedelta(hours=cfg.forecast_horizon_hours)
    if first_origin >= last_origin:
        raise ValueError("Insufficient history for walk-forward origins and forecast horizon")
    candidates = pd.date_range(
        first_origin.ceil(f"{cfg.walk_forward_step_hours}h"),
        last_origin,
        freq=f"{cfg.walk_forward_step_hours}h",
    )
    if len(candidates) > cfg.walk_forward_max_origins:
        candidates = candidates[-cfg.walk_forward_max_origins :]
    rows: list[dict[str, Any]] = []
    for origin in candidates:
        train_mask = timestamps <= origin
        test_mask = (timestamps > origin) & (
            timestamps <= origin + pd.Timedelta(hours=cfg.forecast_horizon_hours)
        )
        if int(train_mask.sum()) < 2 or not test_mask.any():
            continue
        estimator = candidate_estimators(cfg)[model_name]
        estimator.fit(dataset.predictors.loc[train_mask], dataset.target.loc[train_mask])
        prediction = np.asarray(estimator.predict(dataset.predictors.loc[test_mask]), dtype=float)
        metrics = price_metrics(dataset.target.loc[test_mask], prediction)
        rows.append(
            {
                "forecast_origin_utc": origin,
                "training_end_utc": timestamps[train_mask].max(),
                "first_target_utc": timestamps[test_mask].min(),
                "last_target_utc": timestamps[test_mask].max(),
                "target_rows": int(test_mask.sum()),
                "model_name": model_name,
                **metrics,
            }
        )
    if not rows:
        raise ValueError("No valid walk-forward origins")
    result = pd.DataFrame(rows)
    if not (
        (result["training_end_utc"] <= result["forecast_origin_utc"]).all()
        and (result["first_target_utc"] > result["forecast_origin_utc"]).all()
    ):
        raise AssertionError("Walk-forward chronology failed")
    return result
