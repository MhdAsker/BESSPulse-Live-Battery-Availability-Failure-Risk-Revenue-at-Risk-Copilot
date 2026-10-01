"""Typed commercial benchmark outputs and provenance labels."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

DISCLAIMER = "Counterfactual historical simulation, not actual commercial P&L."


class MarketMode(StrEnum):
    HISTORICAL = "historical"
    FORECAST = "forecast"


class CommercialAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    power_derating_impact_eur: float
    energy_capacity_impact_eur: float
    efficiency_impact_eur: float
    availability_impact_eur: float
    interaction_unattributed_eur: float
    total_revenue_at_risk_eur: float
    method: str = "one-factor restoration from current; non-additive and not uniquely causal"
    restoration_order: str = "not applicable: independent one-factor scenarios"
    data_provenance: str = "COUNTERFACTUAL / DERIVED"


class CommercialBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    asset_id: str
    start_timestamp_utc: datetime
    end_timestamp_utc: datetime
    market_mode: MarketMode
    price_source: str
    price_version: str
    reference_power_mw: float
    reference_rated_energy_mwh: float
    reference_usable_energy_mwh: float
    current_min_discharge_power_mw: float
    current_min_charge_power_mw: float
    current_min_usable_energy_mwh: float
    healthy_gross_revenue_eur: float
    healthy_degradation_cost_eur: float
    healthy_net_revenue_eur: float
    current_gross_revenue_eur: float
    current_degradation_cost_eur: float
    current_net_revenue_eur: float
    revenue_at_risk_eur: float
    revenue_at_risk_fraction: float | None
    attribution: CommercialAttribution
    solver_status: str
    solver_name: str
    price_dataset_hash: str
    capability_dataset_hash: str
    optimization_config_hash: str
    calculation_hash: str
    data_provenance: str
    disclaimer: str = DISCLAIMER
    limitations: tuple[str, ...]
