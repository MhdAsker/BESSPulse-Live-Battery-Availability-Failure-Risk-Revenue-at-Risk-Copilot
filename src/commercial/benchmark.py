"""Healthy-versus-current counterfactual commercial benchmark orchestration."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from commercial.attribution import attribute_revenue_at_risk
from commercial.evaluate import calculation_hash, config_hash, frame_hash
from commercial.revenue import combine_dispatch_revenue
from commercial.schemas import (
    DISCLAIMER,
    CommercialBenchmarkResult,
    MarketMode,
)
from optimization.config import DispatchConfig
from optimization.constraints import DispatchCapability, healthy_capability
from optimization.dispatch import DispatchResult, optimize_dispatch, validate_price_intervals


def current_capability_from_availability(
    availability: pd.DataFrame,
    market_timestamps: pd.Series,
    *,
    charge_efficiency: float | pd.Series = 0.96,
    discharge_efficiency: float | pd.Series = 0.96,
    healthy_usable_energy_mwh: float = 32.0,
) -> tuple[DispatchCapability, pd.DataFrame]:
    """Conservatively aggregate 5-minute capability with interval minima."""

    required = {
        "timestamp_utc",
        "available_discharge_power_mw",
        "available_charge_power_mw",
        "available_discharge_energy_mwh",
        "available_charge_energy_mwh",
        "technical_availability",
    }
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"Missing availability fields: {sorted(missing)}")
    market = pd.DataFrame({"timestamp_utc": pd.to_datetime(market_timestamps, utc=True)})
    if len(market) < 2:
        raise ValueError("At least two market timestamps are required")
    dt = market["timestamp_utc"].diff().dropna()
    if dt.nunique() != 1:
        raise ValueError("Market timestamps must be regular before capability alignment")
    interval = dt.iloc[0]
    source = availability.copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    if isinstance(charge_efficiency, pd.Series):
        source["charge_efficiency"] = charge_efficiency.to_numpy(dtype=float)
    else:
        source["charge_efficiency"] = float(charge_efficiency)
    if isinstance(discharge_efficiency, pd.Series):
        source["discharge_efficiency"] = discharge_efficiency.to_numpy(dtype=float)
    else:
        source["discharge_efficiency"] = float(discharge_efficiency)
    rows: list[dict[str, Any]] = []
    for timestamp in market["timestamp_utc"]:
        group = source.loc[
            (source["timestamp_utc"] >= timestamp)
            & (source["timestamp_utc"] < timestamp + interval)
        ]
        if group.empty:
            raise ValueError(f"No capability observations for market interval {timestamp}")
        row = {"timestamp_utc": timestamp}
        for column in required - {"timestamp_utc"}:
            row[column] = float(group[column].min())
        row["charge_efficiency"] = float(group["charge_efficiency"].min())
        row["discharge_efficiency"] = float(group["discharge_efficiency"].min())
        # Reconstruct usable DC window from directional AC quantities. Applying
        # efficiency once in each inverse conversion returns the underlying window.
        discharge_basis = row["available_discharge_energy_mwh"] / row["discharge_efficiency"]
        charge_basis = row["available_charge_energy_mwh"] * row["charge_efficiency"]
        row["usable_energy_capacity_mwh"] = min(
            healthy_usable_energy_mwh, discharge_basis + charge_basis
        )
        rows.append(row)
    aligned = pd.DataFrame(rows)
    capability = DispatchCapability(
        charge_power_mw=aligned["available_charge_power_mw"].to_numpy(dtype=float),
        discharge_power_mw=aligned["available_discharge_power_mw"].to_numpy(dtype=float),
        usable_energy_capacity_mwh=aligned["usable_energy_capacity_mwh"].to_numpy(dtype=float),
        available_charge_energy_mwh=aligned["available_charge_energy_mwh"].to_numpy(dtype=float),
        available_discharge_energy_mwh=aligned["available_discharge_energy_mwh"].to_numpy(
            dtype=float
        ),
        charge_efficiency=aligned["charge_efficiency"].to_numpy(dtype=float),
        discharge_efficiency=aligned["discharge_efficiency"].to_numpy(dtype=float),
        technical_availability=aligned["technical_availability"].to_numpy(dtype=float),
        name="current_estimated_asset",
    )
    return capability, aligned


def benchmark_assets(
    prices: pd.DataFrame,
    current: DispatchCapability,
    *,
    asset_id: str = "BESS-001",
    market_mode: MarketMode = MarketMode.HISTORICAL,
    price_source: str = "REAL",
    price_version: str = "ENTSO-E",
    config: DispatchConfig | None = None,
) -> tuple[CommercialBenchmarkResult, pd.DataFrame, DispatchResult, DispatchResult]:
    cfg = config or DispatchConfig()
    market, dt_hours = validate_price_intervals(prices)
    healthy = healthy_capability(len(market), cfg)
    healthy_result = optimize_dispatch(market, healthy, cfg)
    current_result = optimize_dispatch(market, current, cfg)
    attribution = attribute_revenue_at_risk(market, current, healthy, current_result, cfg)
    revenue_at_risk = healthy_result.net_revenue_eur - current_result.net_revenue_eur
    fraction = (
        revenue_at_risk / abs(healthy_result.net_revenue_eur)
        if abs(healthy_result.net_revenue_eur) > 1e-9
        else None
    )
    price_hash = frame_hash(market[["timestamp_utc", "price_eur_per_mwh"]])
    capability_frame = pd.DataFrame(
        {
            "charge_power_mw": current.charge_power_mw,
            "discharge_power_mw": current.discharge_power_mw,
            "usable_energy_capacity_mwh": current.usable_energy_capacity_mwh,
            "available_charge_energy_mwh": current.available_charge_energy_mwh,
            "available_discharge_energy_mwh": current.available_discharge_energy_mwh,
            "charge_efficiency": current.charge_efficiency,
            "discharge_efficiency": current.discharge_efficiency,
            "technical_availability": current.technical_availability,
        }
    )
    capability_hash = frame_hash(capability_frame)
    cfg_hash = config_hash(cfg)
    calc_hash = calculation_hash(
        {
            "asset_id": asset_id,
            "start": market["timestamp_utc"].min(),
            "end": market["timestamp_utc"].max() + pd.Timedelta(hours=dt_hours),
            "market_mode": market_mode.value,
            "price_version": price_version,
            "price_hash": price_hash,
            "capability_hash": capability_hash,
            "config_hash": cfg_hash,
        }
    )
    provenance = (
        "COUNTERFACTUAL"
        if market_mode == MarketMode.HISTORICAL
        else "COUNTERFACTUAL BASED ON MODEL-PREDICTED PRICES"
    )
    result = CommercialBenchmarkResult(
        asset_id=asset_id,
        start_timestamp_utc=market["timestamp_utc"].min().to_pydatetime(),
        end_timestamp_utc=(
            market["timestamp_utc"].max() + pd.Timedelta(hours=dt_hours)
        ).to_pydatetime(),
        market_mode=market_mode,
        price_source=price_source,
        price_version=price_version,
        reference_power_mw=cfg.healthy_power_mw,
        reference_rated_energy_mwh=cfg.healthy_rated_energy_mwh,
        reference_usable_energy_mwh=cfg.healthy_usable_energy_mwh,
        current_min_discharge_power_mw=float(current.discharge_power_mw.min()),
        current_min_charge_power_mw=float(current.charge_power_mw.min()),
        current_min_usable_energy_mwh=float(current.usable_energy_capacity_mwh.min()),
        healthy_gross_revenue_eur=healthy_result.gross_revenue_eur,
        healthy_degradation_cost_eur=healthy_result.degradation_cost_eur,
        healthy_net_revenue_eur=healthy_result.net_revenue_eur,
        current_gross_revenue_eur=current_result.gross_revenue_eur,
        current_degradation_cost_eur=current_result.degradation_cost_eur,
        current_net_revenue_eur=current_result.net_revenue_eur,
        revenue_at_risk_eur=revenue_at_risk,
        revenue_at_risk_fraction=fraction,
        attribution=attribution,
        solver_status=f"healthy={healthy_result.solver_status};current={current_result.solver_status}",
        solver_name=healthy_result.solver_name,
        price_dataset_hash=price_hash,
        capability_dataset_hash=capability_hash,
        optimization_config_hash=cfg_hash,
        calculation_hash=calc_hash,
        data_provenance=provenance,
        disclaimer=DISCLAIMER,
        limitations=(
            "Historical mode is an ex-post perfect-hindsight counterfactual benchmark."
            if market_mode == MarketMode.HISTORICAL
            else "Forecast mode depends on price-model quality and is not realized future revenue.",
            "Linear throughput degradation cost is a PROJECT ASSUMPTION.",
            "Attribution is order/context dependent and not uniquely causal.",
        ),
    )
    intervals = combine_dispatch_revenue(healthy_result, current_result)
    return result, intervals, healthy_result, current_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run BESSPulse counterfactual commercial benchmark"
    )
    parser.add_argument("--asset-id", default="BESS-001")
    parser.add_argument("--market-mode", choices=[item.value for item in MarketMode], required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/commercial/cli_summary.json"))
    args = parser.parse_args(argv)
    prices = (
        pd.read_parquet(args.prices)
        if args.prices.suffix == ".parquet"
        else pd.read_csv(args.prices)
    )
    availability = pd.read_parquet(args.availability)
    capability, _ = current_capability_from_availability(availability, prices["timestamp_utc"])
    mode = MarketMode(args.market_mode)
    result, _, _, _ = benchmark_assets(
        prices,
        capability,
        asset_id=args.asset_id,
        market_mode=mode,
        price_source="REAL" if mode == MarketMode.HISTORICAL else "MODEL_PREDICTION",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
