"""Transparent one-factor counterfactual restoration attribution."""

from dataclasses import replace

import numpy as np
import pandas as pd

from commercial.schemas import CommercialAttribution
from optimization.config import DispatchConfig
from optimization.constraints import DispatchCapability
from optimization.dispatch import DispatchResult, optimize_dispatch


def _impact(
    prices: pd.DataFrame,
    scenario: DispatchCapability,
    current_revenue: float,
    config: DispatchConfig,
) -> float:
    return optimize_dispatch(prices, scenario, config).net_revenue_eur - current_revenue


def attribute_revenue_at_risk(
    prices: pd.DataFrame,
    current: DispatchCapability,
    healthy: DispatchCapability,
    current_result: DispatchResult,
    config: DispatchConfig,
) -> CommercialAttribution:
    """Restore each factor independently; retain interaction instead of forcing additivity."""

    power_case = replace(
        current,
        charge_power_mw=healthy.charge_power_mw.copy(),
        discharge_power_mw=healthy.discharge_power_mw.copy(),
        name="restore_power",
    )
    energy_case = replace(
        current,
        usable_energy_capacity_mwh=healthy.usable_energy_capacity_mwh.copy(),
        available_charge_energy_mwh=healthy.available_charge_energy_mwh.copy(),
        available_discharge_energy_mwh=healthy.available_discharge_energy_mwh.copy(),
        name="restore_energy",
    )
    efficiency_case = replace(
        current,
        charge_efficiency=healthy.charge_efficiency.copy(),
        discharge_efficiency=healthy.discharge_efficiency.copy(),
        name="restore_efficiency",
    )
    availability = np.maximum(current.technical_availability, 1e-9)
    availability_case = replace(
        current,
        charge_power_mw=np.minimum(healthy.charge_power_mw, current.charge_power_mw / availability),
        discharge_power_mw=np.minimum(
            healthy.discharge_power_mw, current.discharge_power_mw / availability
        ),
        usable_energy_capacity_mwh=np.minimum(
            healthy.usable_energy_capacity_mwh,
            current.usable_energy_capacity_mwh / availability,
        ),
        available_charge_energy_mwh=np.minimum(
            healthy.available_charge_energy_mwh,
            current.available_charge_energy_mwh / availability,
        ),
        available_discharge_energy_mwh=np.minimum(
            healthy.available_discharge_energy_mwh,
            current.available_discharge_energy_mwh / availability,
        ),
        technical_availability=np.ones_like(availability),
        name="restore_availability",
    )
    base = current_result.net_revenue_eur
    power = _impact(prices, power_case, base, config)
    energy = _impact(prices, energy_case, base, config)
    efficiency = _impact(prices, efficiency_case, base, config)
    availability_impact = _impact(prices, availability_case, base, config)
    healthy_revenue = optimize_dispatch(prices, healthy, config).net_revenue_eur
    total = healthy_revenue - base
    interaction = total - power - energy - efficiency - availability_impact
    return CommercialAttribution(
        power_derating_impact_eur=power,
        energy_capacity_impact_eur=energy,
        efficiency_impact_eur=efficiency,
        availability_impact_eur=availability_impact,
        interaction_unattributed_eur=interaction,
        total_revenue_at_risk_eur=total,
    )
