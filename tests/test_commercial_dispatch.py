import numpy as np
import pandas as pd
import pytest

from commercial.benchmark import benchmark_assets
from commercial.schemas import DISCLAIMER, MarketMode
from optimization import DispatchCapability, DispatchConfig, healthy_capability, optimize_dispatch
from optimization.dispatch import validate_price_intervals


def prices(values: list[float], minutes: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2026-01-01", periods=len(values), freq=f"{minutes}min", tz="UTC"
            ),
            "price_eur_per_mwh": values,
        }
    )


def degraded(
    count: int,
    *,
    power: float = 1.5,
    energy: float = 2.4,
    efficiency: float = 0.9,
    availability: float = 0.75,
) -> DispatchCapability:
    ones = np.ones(count)
    return DispatchCapability(
        charge_power_mw=ones * power,
        discharge_power_mw=ones * power,
        usable_energy_capacity_mwh=ones * energy,
        available_charge_energy_mwh=ones * energy / efficiency,
        available_discharge_energy_mwh=ones * energy * efficiency,
        charge_efficiency=ones * efficiency,
        discharge_efficiency=ones * efficiency,
        technical_availability=ones * availability,
        name="degraded",
    )


@pytest.mark.parametrize("minutes,expected", [(15, 0.25), (30, 0.5), (60, 1.0)])
def test_resolution_energy_balance_and_terminal_policy(minutes: int, expected: float) -> None:
    market = prices([-50, -40, 100, 120], minutes)
    cfg = DispatchConfig(healthy_power_mw=2, healthy_rated_energy_mwh=4)
    result = optimize_dispatch(market, healthy_capability(4, cfg), cfg)
    assert result.intervals["dt_hours"].iloc[0] == expected
    assert result.validation["max_energy_balance_error_mwh"] < 1e-6
    assert result.validation["terminal_energy_error_mwh"] < 2e-6
    assert result.intervals["energy_start_mwh"].min() >= -1e-8
    assert result.intervals["energy_end_mwh"].max() <= cfg.healthy_usable_energy_mwh + 1e-8


def test_negative_prices_do_not_enable_simultaneous_cycle() -> None:
    cfg = DispatchConfig(
        healthy_power_mw=2, healthy_rated_energy_mwh=4, degradation_cost_eur_per_mwh=0
    )
    result = optimize_dispatch(prices([-500, -500, 100, 100]), healthy_capability(4, cfg), cfg)
    assert (
        np.minimum(
            result.intervals["charge_power_mw"], result.intervals["discharge_power_mw"]
        ).max()
        == 0
    )
    assert result.intervals.loc[:1, "charge_power_mw"].sum() > 0
    assert result.intervals.loc[2:, "discharge_power_mw"].sum() > 0


def test_constant_price_and_degradation_has_no_free_profit() -> None:
    cfg = DispatchConfig(
        healthy_power_mw=2, healthy_rated_energy_mwh=4, degradation_cost_eur_per_mwh=20
    )
    result = optimize_dispatch(prices([50] * 6), healthy_capability(6, cfg), cfg)
    assert result.net_revenue_eur == pytest.approx(0, abs=5e-5)
    assert result.intervals[["charge_power_mw", "discharge_power_mw"]].to_numpy().max() < 1e-6


def test_degradation_cost_and_efficiency_reduce_cycling_value() -> None:
    market = prices([-20, -10, 100, 120])
    low = DispatchConfig(
        healthy_power_mw=2, healthy_rated_energy_mwh=4, degradation_cost_eur_per_mwh=0
    )
    high = low.model_copy(update={"degradation_cost_eur_per_mwh": 100})
    low_result = optimize_dispatch(market, healthy_capability(4, low), low)
    high_result = optimize_dispatch(market, healthy_capability(4, high), high)
    assert (
        high_result.intervals["charge_energy_mwh"].sum()
        <= low_result.intervals["charge_energy_mwh"].sum() + 1e-6
    )
    inefficient = degraded(
        4, power=2, energy=low.healthy_usable_energy_mwh, efficiency=0.8, availability=1
    )
    assert (
        optimize_dispatch(market, inefficient, low).net_revenue_eur
        <= low_result.net_revenue_eur + 1e-6
    )


def test_current_power_energy_and_offline_limits_bind() -> None:
    market = prices([-100, -100, 500, 500])
    cfg = DispatchConfig(
        healthy_power_mw=2, healthy_rated_energy_mwh=4, degradation_cost_eur_per_mwh=0
    )
    current = degraded(4, power=0.5, energy=0.8, availability=0.25)
    result = optimize_dispatch(market, current, cfg)
    assert result.intervals["charge_power_mw"].max() <= 0.5 + 1e-7
    assert result.intervals["discharge_power_mw"].max() <= 0.5 + 1e-7
    assert result.intervals["energy_end_mwh"].max() <= 0.8 + 1e-7


def test_benchmark_economics_provenance_and_attribution() -> None:
    market = prices([-100, -20, 300, 500])
    cfg = DispatchConfig(healthy_power_mw=2, healthy_rated_energy_mwh=4)
    current = degraded(4)
    original = current.charge_power_mw.copy()
    result, intervals, healthy, degraded_result = benchmark_assets(market, current, config=cfg)
    assert healthy.gross_revenue_eur - healthy.degradation_cost_eur == pytest.approx(
        healthy.net_revenue_eur
    )
    assert result.revenue_at_risk_eur == pytest.approx(
        healthy.net_revenue_eur - degraded_result.net_revenue_eur
    )
    assert result.healthy_net_revenue_eur >= result.current_net_revenue_eur
    assert result.disclaimer == DISCLAIMER
    assert result.data_provenance == "COUNTERFACTUAL"
    assert intervals["interval_revenue_at_risk_eur"].sum() == pytest.approx(
        result.revenue_at_risk_eur
    )
    impacts = result.attribution
    total = (
        impacts.power_derating_impact_eur
        + impacts.energy_capacity_impact_eur
        + impacts.efficiency_impact_eur
        + impacts.availability_impact_eur
        + impacts.interaction_unattributed_eur
    )
    assert total == pytest.approx(result.revenue_at_risk_eur)
    np.testing.assert_array_equal(current.charge_power_mw, original)


def test_forecast_mode_and_irregular_intervals() -> None:
    market = prices([-20, 0, 50, 100])
    result, _, _, _ = benchmark_assets(
        market, degraded(4), market_mode=MarketMode.FORECAST, price_source="MODEL_PREDICTION"
    )
    assert result.price_source == "MODEL_PREDICTION"
    assert result.data_provenance == "COUNTERFACTUAL BASED ON MODEL-PREDICTED PRICES"
    irregular = market.drop(index=2)
    with pytest.raises(ValueError, match="Irregular"):
        validate_price_intervals(irregular)
