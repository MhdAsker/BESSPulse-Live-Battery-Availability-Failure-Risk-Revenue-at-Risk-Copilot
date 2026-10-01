"""Controlled solver boundary and optimization failure contract."""

from typing import Any

import cvxpy as cp


class DispatchOptimizationError(RuntimeError):
    pass


def solve_problem(problem: cp.Problem, solver_name: str) -> tuple[str, str, float, float]:
    if solver_name not in cp.installed_solvers():  # type: ignore[no-untyped-call]
        raise DispatchOptimizationError(f"Configured solver is unavailable: {solver_name}")
    try:
        objective = problem.solve(solver=solver_name, verbose=False)  # type: ignore[no-untyped-call]
    except cp.error.SolverError as exc:
        raise DispatchOptimizationError(f"Dispatch solver failed: {solver_name}") from exc
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise DispatchOptimizationError(f"Dispatch optimization status is {problem.status}")
    stats: Any = problem.solver_stats
    solve_time = float(stats.solve_time) if stats.solve_time is not None else 0.0
    return str(problem.status), solver_name, float(objective), solve_time
