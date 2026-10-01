# Commercial benchmark engine

The engine joins a regular 15/30/60-minute price sequence with conservatively aggregated Prompt 7
capability. Five-minute power and energy limits use the minimum within each market interval. It
never uses fault labels, anomaly scores, SHAP values, or delivery-risk probabilities as physical
constraints.

`optimization.dispatch` builds and validates the HiGHS MILP. `commercial.benchmark` runs healthy
and current cases, `commercial.revenue` retains interval and cumulative cashflows,
`commercial.attribution` runs non-additive restoration scenarios, and `commercial.storage` writes
content-hashed summaries idempotently. Solver status, balance error, exclusivity, bounds, terminal
condition, and objective reconstruction are validated after every solve.

Historical and forecast outputs are separate under `reports/commercial/`. Prompt 7 capability is
simulated/derived; historical prices are REAL; price forecasts are MODEL_PREDICTION; all optimized
dispatch, revenue, RevenueAtRisk, and attribution values are COUNTERFACTUAL.
