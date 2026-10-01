"""Machine-generated historical and forecast commercial benchmark reports."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from besspulse.database import create_schema
from commercial.benchmark import benchmark_assets, current_capability_from_availability
from commercial.schemas import MarketMode
from commercial.storage import persist_commercial_benchmark
from models.price.experiment import load_real_prices
from optimization.config import DispatchConfig


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _replay_capability(targets: pd.Series, source: pd.DataFrame) -> pd.DataFrame:
    target = pd.DatetimeIndex(pd.to_datetime(targets, utc=True))
    start = pd.Timestamp("2026-02-14", tz="UTC")
    day = source.loc[
        (source["timestamp_utc"] >= start) & (source["timestamp_utc"] < start + pd.Timedelta("24h"))
    ].copy()
    if day.empty:
        raise ValueError("Prompt 7 degraded capability day is unavailable")
    target_start = target.min()
    day["timestamp_utc"] = target_start + (day["timestamp_utc"] - start)
    return day


def run_commercial_experiments(
    *,
    database_url: str = "sqlite:///data/besspulse.db",
    report_dir: str | Path = "reports/commercial",
) -> dict[str, Any]:
    output = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = DispatchConfig()
    availability = pd.read_parquet("reports/availability/availability_snapshots.parquet")
    availability["timestamp_utc"] = pd.to_datetime(availability["timestamp_utc"], utc=True)

    real = load_real_prices(database_url).rename(
        columns={"day_ahead_price_eur_per_mwh": "price_eur_per_mwh"}
    )
    start = pd.Timestamp("2024-12-12", tz="UTC")
    historical = real.loc[
        (real["timestamp_utc"] >= start) & (real["timestamp_utc"] < start + pd.Timedelta("24h")),
        ["timestamp_utc", "price_eur_per_mwh"],
    ].copy()
    replay = _replay_capability(historical["timestamp_utc"], availability)
    current, aligned = current_capability_from_availability(
        replay,
        historical["timestamp_utc"],
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        healthy_usable_energy_mwh=cfg.healthy_usable_energy_mwh,
    )
    historical_result, historical_intervals, healthy, current_dispatch = benchmark_assets(
        historical,
        current,
        market_mode=MarketMode.HISTORICAL,
        price_source="REAL",
        price_version="ENTSO-E_DE-LU_PT15M_2024",
        config=cfg,
    )
    _write_json(
        output / "historical_benchmark_summary.json", historical_result.model_dump(mode="json")
    )
    historical_intervals.to_csv(output / "historical_revenue_timeseries.csv", index=False)
    historical_intervals.to_csv(output / "historical_dispatch.csv", index=False)
    pd.DataFrame([historical_result.attribution.model_dump()]).to_csv(
        output / "attribution.csv", index=False
    )
    _write_json(
        output / "optimization_validation.json",
        {"healthy": healthy.validation, "current": current_dispatch.validation},
    )

    forecast_raw = pd.read_parquet("reports/price_forecast/next_24h_forecast.parquet")
    forecast = forecast_raw[["target_timestamp_utc", "predicted_price_eur_per_mwh"]].rename(
        columns={
            "predicted_price_eur_per_mwh": "price_eur_per_mwh",
            "target_timestamp_utc": "timestamp_utc",
        }
    )
    forecast_replay = _replay_capability(forecast["timestamp_utc"], availability)
    forecast_current, _ = current_capability_from_availability(
        forecast_replay,
        forecast["timestamp_utc"],
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        healthy_usable_energy_mwh=cfg.healthy_usable_energy_mwh,
    )
    forecast_result, forecast_intervals, _, _ = benchmark_assets(
        forecast,
        forecast_current,
        market_mode=MarketMode.FORECAST,
        price_source="MODEL_PREDICTION",
        price_version="price_forecast_v1_point",
        config=cfg,
    )
    _write_json(output / "forecast_benchmark_summary.json", forecast_result.model_dump(mode="json"))
    forecast_intervals.to_csv(output / "forecast_dispatch.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "point_selected_model",
                **forecast_result.model_dump(exclude={"attribution", "limitations"}),
            }
        ]
    ).to_csv(output / "forecast_scenarios.csv", index=False)

    create_schema(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        historical_written = persist_commercial_benchmark(
            session, historical_result, capability_version=cfg.capability_version
        )
        forecast_written = persist_commercial_benchmark(
            session, forecast_result, capability_version=cfg.capability_version
        )
    metadata = {
        "historical": historical_result.model_dump(mode="json"),
        "forecast": forecast_result.model_dump(mode="json"),
        "historical_record_written": historical_written,
        "forecast_record_written": forecast_written,
        "capability_replay_source": (
            "Prompt 7 simulated/derived 2026-02-14 profile aligned by time-of-day"
        ),
        "current_efficiency_assumption": "0.95 charge and discharge; PROJECT SCENARIO ASSUMPTION",
        "historical_information_set": "REAL realized prices; ex-post perfect hindsight",
        "forecast_information_set": "Prompt 8 MODEL_PREDICTION point forecast",
        "aligned_capability_rows": len(aligned),
    }
    _write_json(output / "experiment_metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_commercial_experiments(), indent=2, default=str))
