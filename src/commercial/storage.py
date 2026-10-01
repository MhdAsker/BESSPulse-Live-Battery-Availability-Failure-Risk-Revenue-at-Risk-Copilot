"""Idempotent persistence for counterfactual commercial summaries."""

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import RevenueAtRiskModel
from commercial.schemas import CommercialBenchmarkResult


def persist_commercial_benchmark(
    session: Session,
    result: CommercialBenchmarkResult,
    *,
    capability_version: str,
) -> bool:
    existing = session.scalar(
        select(RevenueAtRiskModel.id).where(
            RevenueAtRiskModel.calculation_hash == result.calculation_hash
        )
    )
    if existing is not None:
        return False
    attribution = result.attribution
    session.add(
        RevenueAtRiskModel(
            asset_id=result.asset_id,
            start_timestamp_utc=result.start_timestamp_utc,
            end_timestamp_utc=result.end_timestamp_utc,
            market_mode=result.market_mode.value,
            price_source=result.price_source,
            healthy_gross_revenue_eur=result.healthy_gross_revenue_eur,
            healthy_degradation_cost_eur=result.healthy_degradation_cost_eur,
            healthy_net_revenue_eur=result.healthy_net_revenue_eur,
            current_gross_revenue_eur=result.current_gross_revenue_eur,
            current_degradation_cost_eur=result.current_degradation_cost_eur,
            current_net_revenue_eur=result.current_net_revenue_eur,
            revenue_at_risk_eur=result.revenue_at_risk_eur,
            revenue_at_risk_fraction=result.revenue_at_risk_fraction,
            power_derating_impact_eur=attribution.power_derating_impact_eur,
            energy_capacity_impact_eur=attribution.energy_capacity_impact_eur,
            efficiency_impact_eur=attribution.efficiency_impact_eur,
            availability_impact_eur=attribution.availability_impact_eur,
            interaction_unattributed_eur=attribution.interaction_unattributed_eur,
            model_or_price_version=result.price_version,
            capability_version=capability_version,
            price_dataset_hash=result.price_dataset_hash,
            capability_dataset_hash=result.capability_dataset_hash,
            optimization_config_hash=result.optimization_config_hash,
            calculation_hash=result.calculation_hash,
            attribution_json=json.dumps(attribution.model_dump(mode="json"), sort_keys=True),
            data_provenance=result.data_provenance,
            disclaimer=result.disclaimer,
            created_at=datetime.now(UTC),
        )
    )
    session.commit()
    return True
