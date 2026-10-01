# Revenue at Risk

**Counterfactual historical simulation, not actual commercial P&L.**

BESSPulse compares a healthy 20 MW / 40 MWh reference (32 MWh usable inside configured 10–90%
SOC limits) with the current estimated asset constrained by Prompt 7 directional power, available
energy, charge headroom, component availability, and degraded usable capacity. Both cases see the
same price timestamps, starting usable-energy fraction, terminal rule, and degradation-cost
assumption.

CVXPY maximizes market cashflow minus a configurable linear throughput penalty. Charge and
discharge powers are separate non-negative variables. A binary charging-mode variable and HiGHS
MILP constraints make them mutually exclusive, including at negative prices. For interval length
`dt`, stored usable energy follows `E[t+1] = E[t] + charge[t]*eta_c[t]*dt -
discharge[t]/eta_d[t]*dt - newly_inaccessible_energy[t]`. The last term represents energy becoming
inaccessible when time-varying available capacity falls; it has no market cashflow. State and
directional energy limits remain interval-specific.

Initial usable energy is 50% of each case's available usable window. Terminal energy must return
to the same fraction, within numerical tolerance. The default €2/MWh total charge-plus-discharge
throughput cost is a **PROJECT ASSUMPTION**, not a universal industry value.

Gross revenue is discharge energy times price minus charge energy times price. Net benchmark
revenue subtracts degradation cost. `RevenueAtRisk = healthy net benchmark revenue - current net
benchmark revenue`; it is not clipped. Historical mode uses REAL ENTSO-E prices and is an ex-post,
perfect-hindsight counterfactual. Forecast mode uses Prompt 8 `MODEL_PREDICTION` prices and remains
counterfactual, not future realized revenue.

Attribution independently restores power, energy, efficiency, and availability from the current
case and re-optimizes. These effects overlap and are not uniquely causal; the residual is retained
as `interaction_unattributed_eur`. Every dispatch and revenue output is `COUNTERFACTUAL`.

`GET /api/v1/assets/{asset_id}/revenue-risk` returns only the latest persisted calculation; it does
not launch optimization. The response preserves structured attribution, `COUNTERFACTUAL`
provenance, freshness, and the required disclaimer verbatim.
