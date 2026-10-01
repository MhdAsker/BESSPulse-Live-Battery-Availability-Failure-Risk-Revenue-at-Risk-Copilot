import numpy as np
import pandas as pd
import pytest

from models.price.backtest import walk_forward_backtest
from models.price.baselines import HourOfWeekBaseline, LagBaseline
from models.price.config import PriceModelConfig
from models.price.dataset import TARGET_COLUMN, build_price_dataset
from models.price.evaluate import price_metrics, quantile_metrics
from models.price.quantiles import train_quantile_models
from models.price.train import train_price_model


def observations(periods: int = 1400) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    index = np.arange(periods)
    value = 60 + 18 * np.sin(index * 2 * np.pi / 24) + 8 * np.cos(index * 2 * np.pi / 168)
    return pd.DataFrame({"timestamp_utc": time, TARGET_COLUMN: value})


def test_baselines_are_correct_and_hour_mapping_is_train_only() -> None:
    X = pd.DataFrame(
        {
            "price_lag_24h": [1.0, 2.0],
            "price_lag_168h": [3.0, 4.0],
            "hour_of_week": [1, 2],
        }
    )
    y = pd.Series([10.0, 20.0])
    assert LagBaseline("price_lag_24h").fit(X, y).predict(X).tolist() == [1.0, 2.0]
    assert LagBaseline("price_lag_168h").fit(X, y).predict(X).tolist() == [3.0, 4.0]
    baseline = HourOfWeekBaseline().fit(X, y)
    future = pd.DataFrame({"hour_of_week": [1, 99]})
    assert baseline.predict(future).tolist() == [10.0, 15.0]


def test_metrics_are_exact() -> None:
    metrics = price_metrics(pd.Series([1.0, 3.0]), np.array([2.0, 5.0]))
    assert metrics["mae"] == 1.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["mean_bias_prediction_minus_actual"] == 1.5


def test_lightgbm_reproducible_artifact_and_quantiles() -> None:
    dataset = build_price_dataset(observations())
    first = train_price_model(dataset)
    second = train_price_model(dataset)
    assert first.artifact.model_name == second.artifact.model_name
    np.testing.assert_allclose(
        first.artifact.estimator.predict(first.split.test.predictors),
        second.artifact.estimator.predict(second.split.test.predictors),
    )
    quantiles = train_quantile_models(dataset, split=first.split)
    assert set(quantiles.estimators) == {0.1, 0.5, 0.9}
    assert "p10_gt_p50_fraction" in quantiles.metrics
    metrics = quantile_metrics(
        pd.Series([0.0]), {0.1: np.array([-1.0]), 0.5: np.array([0.0]), 0.9: np.array([1.0])}
    )
    assert metrics["p50_empirical_coverage"] == 1.0


def test_walk_forward_is_strictly_causal() -> None:
    config = PriceModelConfig(walk_forward_max_origins=3)
    dataset = build_price_dataset(observations(), config)
    result = walk_forward_backtest(dataset, "lag_24h", config)
    assert (result["training_end_utc"] <= result["forecast_origin_utc"]).all()
    assert (result["first_target_utc"] > result["forecast_origin_utc"]).all()
