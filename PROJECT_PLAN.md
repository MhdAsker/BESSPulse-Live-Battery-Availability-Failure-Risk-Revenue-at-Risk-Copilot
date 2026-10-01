# Project plan

## Phase 1 - implemented

- Typed configuration, 20 MW / 40 MWh site -> PCS -> rack simulator, ten fault families, isolated fault truth, SQLAlchemy persistence, and behavioral tests.

## Phase 2 / Prompt 2 - implemented

- Secure ENTSO-E DE-LU acquisition, immutable raw XML, namespace-safe parsing, UTC normalization, provenance-aware idempotent storage, and offline integration tests.

## Phase 3 / Prompt 3 - implemented

- Causal site/rack/market features, leave-one-out peer analytics, backward-only market alignment, leakage guards, and content-addressed Parquet snapshots.

## Phase 4 / Prompt 4 - implemented

- Healthy-cohort expected site-power and rack-temperature datasets.
- Chronological 60/20/20 splitting with event-boundary protection and train-only preprocessing.
- Naive, Ridge, Random Forest, and LightGBM validation comparisons with simplicity-aware selection.
- Target-free batch inference, direction-aware power and thermal residuals, and causal persistence features.
- Leakage-safe expanding-window historical prediction with minimum-history enforcement.
- Versioned artifacts, exact feature contracts, dataset hashes, provenance-aware persistence, fault diagnostics, and unseen-rack evaluation.

## Phase 5 / Prompt 5 - implemented

- Telemetry-derived, censored 6/12/24-hour delivery-failure targets.
- Horizon-specific purged train/validation/calibration/test datasets.
- Logistic Regression, Random Forest, LightGBM, and XGBoost plus two baselines.
- Validation-only selection, held-out calibration, isolated testing, and reliability diagnostics.
- Residual/peer/market ablations, consistency/severity diagnostics, persistence, and SHAP.

## Phase 6 / Prompt 6 - implemented

- Engineering, robust peer, Isolation Forest, residual, and transparent vote detectors.
- Validation-only threshold/persistence selection and chronological event evaluation.
- Event merging/matching, false alerts per healthy day, delay, family/severity diagnostics.
- Held-out-rack/unseen-severity diagnostics and matched residual/peer ablations.
- Versioned artifacts, event storage, site summary, optional plots, and descriptive-only delivery-risk comparison.

Isolation Forest's measured false-alert rate is too high for alerting, robust peer detection was ineffective under identical-peer zero-MAD behavior, and no detector demonstrated pre-fault warning. Prompt 5's 24-hour calibration limitation remains unresolved and explicit.

## Phase 7 / Prompt 7 - implemented

- Rack, PCS, and site technical availability with explicit unknown state.
- Directional charge/discharge MW and AC-facing deliverable/acceptable MWh.
- Requested-power availability, one-hour energy sufficiency, and 15m/1h/2h sustainable power.
- Observable limiting factors/components, downtime and derating events, and time-weighted summaries.
- Idempotent `availability_snapshots` persistence and no-ML CLI/service contracts.
- Descriptive-only anomaly/delivery-risk/fault diagnostics after deterministic calculation.

## Later phases

1. Price forecasting and commercial value/revenue-at-risk using the validated directional capability contract.
2. Dispatch optimization only after forecast and constraint validation.
3. Prediction services, monitoring, API/dashboard, and later evidence-grounded agents/RAG.

Every phase must preserve event time, source provenance, scenario/config version, and separation between observables, future outcomes, and evaluation-only truth.
## Prompt 8 — complete

Implemented timestamp-causal DE-LU price baselines, LightGBM point and quantile forecasting,
chronological selection/test evaluation, expanding-window backtesting, artifacts, idempotent
forecast storage, empirical REAL ENTSO-E reporting, and leakage/DST/storage tests. Prompt 9 should
consume central and uncertainty scenarios for dispatch economics while preserving price-origin and
technical-availability constraints; it should not retrain or reinterpret this model implicitly.
## Prompt 9 — recovered and complete

Commercial dispatch, healthy/current benchmarking, RevenueAtRisk, restoration attribution,
content hashes, idempotent storage, historical/forecast reports, tests, and documentation are in
place. Prompt 10 may consume commercial outputs as context without turning them into physical
constraints.
## Prompt 10 — complete

Implemented deterministic priority components and bands, 24h reliability treatment, technical
override, evidence, alert families, deduplication, hysteresis, OPEN/RESOLVED lifecycle, storage,
historical replay, post-generation ground-truth evaluation, reports, tests, and documentation.

## Prompt 11 — complete

Implemented the versioned FastAPI application, typed contracts, request-scoped database sessions,
thin query services, structured provenance and freshness, secret-safe errors, request IDs,
local-only CORS defaults, bounded filtering/pagination, guarded simulation controls, honest Copilot
503 behavior, OpenAPI, integration tests, query indexes, and API documentation. Dashboard,
authentication, LLM agents/RAG, model monitoring, Docker, and deployment remain incomplete.

## Prompt 12 — complete

Implemented the API-first Streamlit dashboard with ten cohesive pages, industrial energy theme,
typed API client, explicit local artifact demo adapter, provenance and freshness UX, battery and
power-flow animation, 32-rack grouped heatmap, risk/calibration and commercial disclosures, alert
console, honest Copilot and model-monitoring shells, auto-refresh/timezone/session state, tests,
documentation, and live runtime smoke verification. Agent reasoning, RAG, monitoring services,
authentication, Docker, and deployment remain later work.
