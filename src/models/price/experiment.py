"""Reproducible REAL ENTSO-E DE-LU price experiment and report generation."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from besspulse.database import MarketDataModel, create_schema
from data.parser import parse_duration
from data.schemas import MarketMetric, MarketObservation, SeriesType
from data.validation import validate_series_continuity
from models.artifact import ModelArtifact, save_model_artifact
from models.price.backtest import walk_forward_backtest
from models.price.config import PriceModelConfig
from models.price.dataset import (
    PRICE_FEATURES,
    TARGET_COLUMN,
    build_price_dataset,
    coverage_report,
    infer_resolution,
)
from models.price.evaluate import price_metrics
from models.price.predict import predict_prices
from models.price.quantiles import train_quantile_models
from models.price.storage import store_price_predictions
from models.price.train import lightgbm_regressor, train_price_model


def load_real_prices(database_url: str) -> pd.DataFrame:
    """Load one latest-retrieved REAL day-ahead observation per delivery timestamp."""

    engine = create_engine(database_url)
    with Session(engine) as session:
        rows = session.scalars(
            select(MarketDataModel)
            .where(
                MarketDataModel.market_region == "DE-LU",
                MarketDataModel.metric == "day_ahead_price",
                MarketDataModel.data_provenance == "REAL",
            )
            .order_by(MarketDataModel.timestamp_utc, MarketDataModel.retrieved_at_utc)
        ).all()
    if not rows:
        raise ValueError("No REAL ENTSO-E DE-LU day-ahead prices are stored")
    frame = pd.DataFrame(
        {
            "timestamp_utc": [row.timestamp_utc for row in rows],
            TARGET_COLUMN: [row.value for row in rows],
            "resolution": [row.resolution for row in rows],
            "retrieved_at_utc": [row.retrieved_at_utc for row in rows],
        }
    )
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["retrieved_at_utc"] = pd.to_datetime(frame["retrieved_at_utc"], utc=True)
    # ENTSO-E can return parallel classification sequences at different resolutions.
    # Use one coherent finest-resolution series rather than mixing observations at
    # coincident timestamps. The chosen resolution is retained in coverage metadata.
    resolutions = sorted(frame["resolution"].dropna().unique(), key=parse_duration)
    selected_resolution = resolutions[0]
    result = (
        frame.loc[frame["resolution"] == selected_resolution]
        .sort_values(["timestamp_utc", "retrieved_at_utc"], kind="mergesort")
        .drop_duplicates("timestamp_utc", keep="last")
        .drop(columns="retrieved_at_utc")
        .reset_index(drop=True)
    )
    return cast(pd.DataFrame, result)


def _continuity(prices: pd.DataFrame) -> dict[str, Any]:
    now = datetime.now(UTC)
    records = cast(list[dict[str, Any]], prices.to_dict(orient="records"))
    observations = tuple(
        MarketObservation(
            timestamp_utc=pd.Timestamp(row["timestamp_utc"]).to_pydatetime(),
            market_region="DE-LU",
            metric=MarketMetric.DAY_AHEAD_PRICE,
            value=float(row[TARGET_COLUMN]),
            unit="EUR/MWh",
            series_type=SeriesType.DAY_AHEAD,
            retrieved_at_utc=now,
            raw_content_hash="0" * 64,
            resolution=str(row["resolution"]),
        )
        for row in records
    )
    report = validate_series_continuity(observations)
    return {
        "expected_interval_minutes": int(report.expected_interval.total_seconds() / 60),
        "expected_observation_count": report.expected_observation_count,
        "actual_observation_count": report.actual_observation_count,
        "missing_timestamps": [value.isoformat() for value in report.missing_timestamps],
        "duplicate_timestamps": [value.isoformat() for value in report.duplicate_timestamps],
    }


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _ablation(dataset: Any, result: Any, cfg: PriceModelConfig) -> pd.DataFrame:
    split = result.split
    groups = {
        "A_calendar_only": [
            "hour_of_day",
            "day_of_week",
            "hour_of_week",
            "is_weekend",
            "utc_offset_hours",
        ],
        "B_price_lags_plus_calendar": [
            "price_lag_24h",
            "price_lag_168h",
            "hour_of_day",
            "day_of_week",
            "hour_of_week",
            "is_weekend",
            "utc_offset_hours",
        ],
        "C_plus_rolling_price": list(PRICE_FEATURES),
    }
    rows: list[dict[str, Any]] = []
    for name, features in groups.items():
        estimator: LGBMRegressor = lightgbm_regressor(cfg)
        estimator.fit(split.train.predictors[features], split.train.target)
        prediction = np.asarray(
            estimator.predict(split.validation.predictors[features]), dtype=float
        )
        rows.append(
            {
                "ablation": name,
                "features": "|".join(features),
                **price_metrics(split.validation.target, prediction),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    database_url: str = "sqlite:///data/besspulse.db",
    *,
    report_dir: str | Path = "reports/price_forecast",
    artifact_dir: str | Path = "artifacts/models/price",
    config: PriceModelConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PriceModelConfig()
    reports = Path(report_dir)
    artifacts = Path(artifact_dir)
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    observations = load_real_prices(database_url)
    coverage = coverage_report(observations, cfg)
    continuity = _continuity(observations)
    dataset = build_price_dataset(observations, cfg)
    result = train_price_model(dataset, cfg)
    comparison = pd.DataFrame.from_dict(result.validation_comparison, orient="index")
    comparison.index.name = "model_name"
    comparison.reset_index().to_csv(reports / "baseline_comparison.csv", index=False)
    comparison.reset_index().to_csv(reports / "model_comparison.csv", index=False)
    _json(reports / "test_metrics.json", result.test_metrics)
    _json(
        reports / "negative_price_metrics.json",
        {
            key: value
            for key, value in result.test_regime_metrics.items()
            if key.startswith("negative")
        },
    )
    _json(
        reports / "high_price_metrics.json",
        {key: value for key, value in result.test_regime_metrics.items() if key.startswith("high")},
    )
    ablation = _ablation(dataset, result, cfg)
    ablation.to_csv(reports / "feature_ablation.csv", index=False)
    walk_forward = walk_forward_backtest(dataset, result.artifact.model_name, cfg)
    walk_forward.to_csv(reports / "walk_forward_metrics.csv", index=False)
    model_path, metadata_path = save_model_artifact(result.artifact, artifacts / cfg.model_version)

    quantile_summary: dict[str, Any] | None = None
    if cfg.quantiles_enabled:
        quantiles = train_quantile_models(dataset, cfg, result.split)
        quantile_summary = quantiles.metrics
        _json(reports / "quantile_metrics.json", quantile_summary)
        for level, estimator in quantiles.estimators.items():
            label = f"p{int(level * 100):02d}"
            artifact = ModelArtifact(
                estimator=estimator,
                feature_names=tuple(dataset.predictors.columns),
                numeric_features=tuple(dataset.predictors.columns),
                categorical_features=(),
                model_name="lightgbm_quantile",
                model_version=f"price_forecast_{label}_v1",
                target_name=cfg.target_name,
                metadata={
                    **result.artifact.metadata,
                    "quantile": level,
                    "model_version": f"price_forecast_{label}_v1",
                    "quantile_postprocessing": "row-wise sort at grouped inference",
                },
            )
            save_model_artifact(artifact, artifacts / artifact.model_version)

    resolution = infer_resolution(observations)
    origin = observations["timestamp_utc"].max()
    periods = int(np.ceil(pd.Timedelta(hours=cfg.forecast_horizon_hours) / resolution))
    targets = pd.date_range(origin + resolution, periods=periods, freq=resolution)
    forecast = predict_prices(result.artifact, observations, targets, origin, config=cfg)
    forecast.to_parquet(reports / "next_24h_forecast.parquet", index=False)
    create_schema(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        stored_forecast_rows = store_price_predictions(session, forecast)
    metadata = {
        "experiment_created_at_utc": datetime.now(UTC).isoformat(),
        "data_provenance": "REAL ENTSO-E DATA",
        "prediction_provenance": "MODEL_PREDICTION",
        "coverage": coverage,
        "continuity_validation": continuity,
        "selected_model": result.artifact.model_name,
        "selection_policy": result.artifact.metadata["selection_policy"],
        "validation_comparison": result.validation_comparison,
        "test_metrics": result.test_metrics,
        "regime_metrics": result.test_regime_metrics,
        "walk_forward_mean_metrics": {
            column: float(walk_forward[column].mean())
            for column in ["mae", "rmse", "r2", "mean_bias_prediction_minus_actual"]
        },
        "feature_ablation": ablation.to_dict(orient="records"),
        "system_forecast_ablation": "not run: only REAL prices available in experiment dataset",
        "quantile_metrics": quantile_summary,
        "forecast_rows": len(forecast),
        "forecast_rows_newly_stored": stored_forecast_rows,
        "forecast_origin_utc": origin.isoformat(),
        "forecast_end_utc": forecast["target_timestamp_utc"].max().isoformat(),
        "model_artifact": str(model_path),
        "model_metadata": str(metadata_path),
    }
    _json(reports / "experiment_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run REAL ENTSO-E price forecasting experiment")
    parser.add_argument("--database-url", default="sqlite:///data/besspulse.db")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.database_url), indent=2, default=str))


if __name__ == "__main__":
    main()
