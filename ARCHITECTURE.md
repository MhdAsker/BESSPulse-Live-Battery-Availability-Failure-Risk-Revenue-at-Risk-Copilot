# Architecture

## Implemented data flow

```text
typed YAML config -> site / PCS / rack simulator -> immutable telemetry
                               |
                      separate fault ground truth

ENTSO-E HTTP -> immutable raw XML -> parser / validator -> market records

telemetry + market records -> causal feature modules -> versioned Parquet snapshots
                                                        |
                              chronological expected-behavior training
                                 |                          |
                       expected site power       expected rack temperature
                                 |                          |
                         predictions + causal actual-minus-expected residuals
                                                        |
                           curated site-level delivery-risk features
                                                        |
                    censored targets -> horizon-specific purged splits
                                                        |
                   train -> validation selection -> calibration -> test

telemetry + peer features + leakage-safe residuals
                 -> healthy-train references / Isolation Forest
                 -> validation-only threshold + causal persistence
                 -> rack anomaly intervals -> merged component events
                 -> evaluation-only matching + transparent site summary

rack telemetry -> rack directional capability -> PCS converter-constrained capability
               -> site technical / power / energy availability
               -> requested-power sufficiency -> historical metrics and events
```

`besspulse.config` owns normal system parameters; environment variables are reserved for secrets and the database URL. Stateful rack physics, PCS aggregation, and site request allocation remain independent of persistence. Positive power is discharge and negative power is charge. Telemetry schemas contain observables only, while `ground_truth.py` owns fault schedules and labels.

`data.entsoe` separates request construction, bounded retries, immutable raw storage, XML parsing, continuity checks, and normalized persistence. A content hash links each real normalized observation to its source document. Feature modules reject duplicate identities, naive timestamps, leakage columns, and infinities. Rolling calculations are causal, peers are leave-one-out, and market joins are backward-only with finite tolerance.

`models.expected_power` and `models.expected_temperature` separate dataset construction, training, inference, evaluation, and artifacts. Shared model code enforces unique-timestamp chronological splitting, fault-event boundary protection, healthy-cohort training, train-only preprocessing, deterministic candidates, exact inference contracts, and expanding-window historical predictions. Ground truth can select healthy offline rows and annotate diagnostics but never enters `X`.

`ModelPredictionModel` stores optional actuals and residuals beside predictions while preserving distinct `MODEL_PREDICTION` and `DERIVED` provenance. Artifacts are local joblib pipelines plus JSON metadata; only trusted project artifacts may be loaded. `dashboard/` remains a placeholder.

`models.delivery_risk` isolates telemetry-derived targets from causal features, builds independent 6/12/24-hour datasets, purges target-window overlap at every partition boundary, fits preprocessing on train only, selects classifiers on validation, calibrates on a later block, and evaluates test once. Probability artifacts keep horizon-specific contracts and are never substituted across horizons.

`models.anomaly` separates engineering, robust peer, Isolation Forest, residual, ensemble, event construction, evaluation, artifact, storage, and plotting concerns. Ground truth is cohort/evaluation metadata only. Rack evidence remains primary; site summaries expose counts/maxima rather than an opaque priority score. `anomaly_events` stores component events without commercial priority fields.

`availability` is deterministic and has no ML dependency. It retains charge/discharge direction, installed-denominator unknown handling, PCS/rack conservation, and AC-facing energy semantics. Delivery-risk and anomaly outputs are adjacent contextual joins only and cannot change capability. `availability_snapshots` is unique by asset and timestamp.

## Deliberate decisions

- Half-open fault windows `[start, end)` remove boundary ambiguity.
- Site requests are allocated by current component capability.
- Expected power uses one model with non-ordinal operating mode rather than separate charge/discharge models.
- Rack and PCS identities are metadata, not temperature predictors.
- Selection uses validation RMSE and chooses the simplest candidate within a fixed 2% tolerance.
- Test data are evaluated after selection and never influence the chosen model.
- Historical residual features use only models trained on strictly earlier healthy rows.
## Price forecasting boundary

`models.price.dataset` owns causal UTC feature construction and the price-specific leakage guard;
`baselines`, `train`, `quantiles`, and `evaluate` own modeling; `predict` owns origin-aware output;
`backtest` owns expanding-window replay; and `storage` owns idempotent `price_predictions` writes.
The package consumes normalized ENTSO-E observations but never imports battery, availability,
anomaly, or delivery-risk features. Artifacts use the existing trusted-local model contract.
## Commercial flow

`REAL / MODEL_PREDICTION price + DERIVED deterministic availability → CVXPY/HiGHS dispatch →
healthy and current COUNTERFACTUAL benchmarks → RevenueAtRisk → non-causal restoration
attribution`. Risk and anomaly outputs remain context outside physical constraints.
## Alert flow

`telemetry/features → expected behavior + delivery risk + anomaly + availability + commercial
benchmark → structured evidence → transparent priority → lifecycle alert`. Ground truth is outside
generation and enters only replay evaluation.

## API delivery layer

```text
database + existing domain services
                 |
        API-facing query services
                 |
      typed FastAPI /api/v1 routes
                 |
       future dashboard / agents
```

FastAPI is a delivery layer. Request-scoped sessions perform bounded latest-row and event queries;
application lifespan owns reusable simulation/Copilot services. Routes contain no scientific or ML
calculations. GET revenue risk returns the latest persisted benchmark and cannot invoke CVXPY.
Missing optional analytics degrade individual endpoints, while Copilot absence does not make the
core API unready. Composite indexes support asset/rack time queries and alert filtering.

## Dashboard delivery layer

```text
domain / persistence → FastAPI typed contracts → dashboard API client → Streamlit pages
                                                ↘ explicit local artifact demo adapter
```

The normal path is API-driven. Streamlit owns presentation, navigation, caching, formatting,
timezone display, and session state only. The isolated demo adapter selects existing persisted
reports for historical charts and the rack grid when no matching aggregate API exists; it is
visibly labeled and cannot train, infer, optimize, calculate availability, or prioritize alerts.
Reusable components centralize theme, Plotly layout, provenance badges, cards, battery/power-flow
animation, grouped rack tiles, status strips, loading, and empty/error states.
