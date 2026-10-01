"""CVXPY MILP dispatch with explicit charge/discharge modes and validation."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization.config import DispatchConfig
from optimization.constraints import DispatchCapability
from optimization.solver import DispatchOptimizationError, solve_problem


@dataclass(frozen=True)
class DispatchResult:
    asset_case: str
    intervals: pd.DataFrame
    gross_revenue_eur: float
    degradation_cost_eur: float
    net_revenue_eur: float
    objective_value: float
    solver_status: str
    solver_name: str
    solve_time_seconds: float
    validation: dict[str, float | bool]
    data_provenance: str = "COUNTERFACTUAL"


def validate_price_intervals(prices: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    required = {"timestamp_utc", "price_eur_per_mwh"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price fields: {sorted(missing)}")
    frame = prices.copy().sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    raw = frame["timestamp_utc"]
    for value in raw:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Market timestamps must be timezone-aware")
    frame["timestamp_utc"] = pd.to_datetime(raw, utc=True)
    if frame["timestamp_utc"].duplicated().any() or len(frame) < 2:
        raise ValueError("Market timestamps must be unique with at least two intervals")
    differences = frame["timestamp_utc"].diff().dropna()
    interval = pd.Timedelta(differences.iloc[0])
    if differences.nunique() != 1 or interval <= pd.Timedelta(0):
        raise ValueError("Irregular or missing market intervals are not supported")
    values = pd.to_numeric(frame["price_eur_per_mwh"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prices must be finite")
    frame["price_eur_per_mwh"] = values
    return frame, float(interval.total_seconds() / 3600)


def optimize_dispatch(
    prices: pd.DataFrame,
    capability: DispatchCapability,
    config: DispatchConfig | None = None,
) -> DispatchResult:
    cfg = config or DispatchConfig()
    market, dt_hours = validate_price_intervals(prices)
    count = len(market)
    capability.validate(count)
    price = market["price_eur_per_mwh"].to_numpy(dtype=float)

    charge = cp.Variable(count, nonneg=True, name="charge_power_mw")
    discharge = cp.Variable(count, nonneg=True, name="discharge_power_mw")
    energy = cp.Variable(count + 1, nonneg=True, name="usable_energy_mwh")
    inaccessible = cp.Variable(count, nonneg=True, name="newly_inaccessible_energy_mwh")
    charging_mode = cp.Variable(count, boolean=True, name="charging_mode")

    initial_energy = cfg.initial_soc_fraction * capability.usable_energy_capacity_mwh[0]
    terminal_capacity = capability.usable_energy_capacity_mwh[-1]
    constraints: list[cp.Constraint] = [energy[0] == initial_energy]
    constraints.extend(
        [
            charge <= cp.multiply(capability.charge_power_mw, charging_mode),  # type: ignore[attr-defined]
            discharge <= cp.multiply(capability.discharge_power_mw, 1 - charging_mode),  # type: ignore[attr-defined]
            cp.multiply(charge, dt_hours) <= capability.available_charge_energy_mwh,  # type: ignore[attr-defined]
            cp.multiply(discharge, dt_hours) <= capability.available_discharge_energy_mwh,  # type: ignore[attr-defined]
        ]
    )
    for index in range(count):
        next_capacity = (
            capability.usable_energy_capacity_mwh[index + 1]
            if index + 1 < count
            else terminal_capacity
        )
        constraints.extend(
            [
                energy[index] <= capability.usable_energy_capacity_mwh[index],
                energy[index + 1] <= next_capacity,
                energy[index + 1]
                == energy[index]
                + charge[index] * capability.charge_efficiency[index] * dt_hours
                - discharge[index] / capability.discharge_efficiency[index] * dt_hours
                - inaccessible[index],
            ]
        )
    terminal_target = cfg.initial_soc_fraction * terminal_capacity
    if cfg.terminal_policy == "equal_initial_fraction":
        constraints.extend(
            [
                energy[-1] >= terminal_target - cfg.terminal_tolerance_mwh,
                energy[-1] <= terminal_target + cfg.terminal_tolerance_mwh,
            ]
        )
    else:
        constraints.append(energy[-1] >= terminal_target - cfg.terminal_tolerance_mwh)

    gross_expression = cp.sum(cp.multiply(discharge - charge, price)) * dt_hours  # type: ignore[attr-defined]
    throughput_expression = cp.sum(charge + discharge) * dt_hours  # type: ignore[attr-defined]
    degradation_expression = throughput_expression * cfg.degradation_cost_eur_per_mwh
    problem = cp.Problem(cp.Maximize(gross_expression - degradation_expression), constraints)
    status, solver_name, objective, solve_time = solve_problem(problem, cfg.solver_name)
    if any(value.value is None for value in [charge, discharge, energy, inaccessible]):
        raise DispatchOptimizationError("Solver returned no dispatch values")

    charge_value = np.maximum(np.asarray(charge.value, dtype=float), 0)
    discharge_value = np.maximum(np.asarray(discharge.value, dtype=float), 0)
    energy_value = np.maximum(np.asarray(energy.value, dtype=float), 0)
    inaccessible_value = np.maximum(np.asarray(inaccessible.value, dtype=float), 0)
    charge_mwh = charge_value * dt_hours
    discharge_mwh = discharge_value * dt_hours
    gross_cashflow = (discharge_mwh - charge_mwh) * price
    degradation = (charge_mwh + discharge_mwh) * cfg.degradation_cost_eur_per_mwh
    net_cashflow = gross_cashflow - degradation
    frame = market.copy()
    frame["dt_hours"] = dt_hours
    frame["charge_power_mw"] = charge_value
    frame["discharge_power_mw"] = discharge_value
    frame["net_dispatch_power_mw"] = discharge_value - charge_value
    frame["charge_energy_mwh"] = charge_mwh
    frame["discharge_energy_mwh"] = discharge_mwh
    frame["energy_start_mwh"] = energy_value[:-1]
    frame["energy_end_mwh"] = energy_value[1:]
    frame["newly_inaccessible_energy_mwh"] = inaccessible_value
    frame["gross_cashflow_eur"] = gross_cashflow
    frame["degradation_cost_eur"] = degradation
    frame["net_cashflow_eur"] = net_cashflow
    frame["data_provenance"] = "COUNTERFACTUAL"

    balance_rhs = (
        energy_value[:-1]
        + charge_value * capability.charge_efficiency * dt_hours
        - discharge_value / capability.discharge_efficiency * dt_hours
        - inaccessible_value
    )
    max_balance_error = float(np.max(np.abs(energy_value[1:] - balance_rhs)))
    simultaneous = np.minimum(charge_value, discharge_value)
    reconstructed = float(net_cashflow.sum())
    validation: dict[str, float | bool] = {
        "max_energy_balance_error_mwh": max_balance_error,
        "max_simultaneous_charge_discharge_mw": float(simultaneous.max()),
        "max_charge_limit_violation_mw": float(
            np.maximum(charge_value - capability.charge_power_mw, 0).max()
        ),
        "max_discharge_limit_violation_mw": float(
            np.maximum(discharge_value - capability.discharge_power_mw, 0).max()
        ),
        "terminal_energy_error_mwh": float(abs(energy_value[-1] - terminal_target)),
        "objective_reconstruction_error_eur": float(abs(reconstructed - objective)),
        "all_checks_passed": bool(
            max_balance_error <= cfg.feasibility_tolerance
            and simultaneous.max() <= cfg.simultaneous_power_tolerance_mw
            and abs(energy_value[-1] - terminal_target)
            <= cfg.terminal_tolerance_mwh + cfg.feasibility_tolerance
            and abs(reconstructed - objective) <= 1e-3
        ),
    }
    if not validation["all_checks_passed"]:
        raise DispatchOptimizationError(f"Post-solve validation failed: {validation}")
    return DispatchResult(
        asset_case=capability.name,
        intervals=frame,
        gross_revenue_eur=float(gross_cashflow.sum()),
        degradation_cost_eur=float(degradation.sum()),
        net_revenue_eur=float(net_cashflow.sum()),
        objective_value=objective,
        solver_status=status,
        solver_name=solver_name,
        solve_time_seconds=solve_time,
        validation=validation,
    )
