"""Persisted Revenue-at-Risk response contracts."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, Freshness, Provenance


class CommercialAttributionResponse(APIModel):
    power_derating_impact_eur: float
    energy_capacity_impact_eur: float
    efficiency_impact_eur: float
    availability_impact_eur: float
    interaction_unattributed_eur: float


class RevenueRiskResponse(APIModel):
    asset_id: str
    start_timestamp_utc: datetime
    end_timestamp_utc: datetime
    market_mode: str
    price_source: str
    healthy_gross_revenue_eur: float
    healthy_degradation_cost_eur: float = Field(ge=0)
    healthy_net_revenue_eur: float
    current_gross_revenue_eur: float
    current_degradation_cost_eur: float = Field(ge=0)
    current_net_revenue_eur: float
    revenue_at_risk_eur: float
    revenue_at_risk_fraction: float | None
    attribution: CommercialAttributionResponse
    calculation_version: str
    freshness: Freshness
    data_provenance: Provenance
    disclaimer: str
