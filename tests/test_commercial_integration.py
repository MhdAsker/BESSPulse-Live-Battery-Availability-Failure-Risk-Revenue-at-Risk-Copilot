import json

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from besspulse.database import Base, RevenueAtRiskModel
from commercial.benchmark import benchmark_assets, current_capability_from_availability
from commercial.storage import persist_commercial_benchmark
from optimization.config import DispatchConfig


def test_end_to_end_capability_benchmark_and_idempotent_storage() -> None:
    market_time = pd.Series(pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"))
    telemetry_time = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    availability = pd.DataFrame(
        {
            "timestamp_utc": telemetry_time,
            "available_discharge_power_mw": [20] * 3 + [15] * 9,
            "available_charge_power_mw": [20] * 3 + [15] * 9,
            "available_discharge_energy_mwh": [15] * 12,
            "available_charge_energy_mwh": [17] * 12,
            "technical_availability": [1] * 3 + [0.75] * 9,
        }
    )
    capability, aligned = current_capability_from_availability(availability, market_time)
    assert aligned.loc[1, "available_discharge_power_mw"] == 15
    prices = pd.DataFrame({"timestamp_utc": market_time, "price_eur_per_mwh": [-50, -20, 100, 200]})
    config = DispatchConfig()
    result, _, _, _ = benchmark_assets(prices, capability, config=config)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert persist_commercial_benchmark(
            session, result, capability_version=config.capability_version
        )
        assert not persist_commercial_benchmark(
            session, result, capability_version=config.capability_version
        )
        stored = session.scalar(select(RevenueAtRiskModel))
    assert stored is not None
    assert stored.price_dataset_hash == result.price_dataset_hash
    assert stored.capability_dataset_hash == result.capability_dataset_hash
    assert stored.optimization_config_hash == result.optimization_config_hash
    assert json.loads(stored.attribution_json)["method"].startswith("one-factor")
    assert stored.data_provenance == "COUNTERFACTUAL"
