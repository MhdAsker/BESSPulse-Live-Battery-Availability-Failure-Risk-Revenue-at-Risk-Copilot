# BESSPulse

BESSPulse is a provenance-aware research system for studying operational underperformance in grid-scale battery energy storage. It combines a configurable 20 MW / 40 MWh reduced-order digital twin, ENTSO-E DE-LU market ingestion, causal feature engineering, expected-behavior models, requested-power delivery-risk classification, and component anomaly detection.

## Current scope

Implemented capabilities include:

- A site -> PCS -> rack simulator with ten deterministic fault families.
- Immutable observable telemetry and separately stored synthetic fault ground truth.
- Secure ENTSO-E ingestion, raw XML retention, UTC normalization, and idempotent persistence.
- Causal site, rack, market, and leave-one-out peer features with content-addressed snapshots.
- Healthy-behavior expected site-power and rack-temperature regressors.
- Naive, Ridge, Random Forest, and LightGBM comparison using chronological validation only.
- Target-free inference, provenance-aware storage, causal residual persistence, and expanding-window historical residuals.
- Censored 6/12/24-hour delivery-failure targets with horizon-specific purge gaps.
- Logistic Regression, Random Forest, LightGBM, and XGBoost classification, calibration, ablations, and SHAP explanations.
- Engineering, robust-peer, Isolation Forest, residual, and transparent vote anomaly detectors with interval-to-event evaluation.
- Deterministic rack/PCS/site availability with separate technical, directional power, directional energy, and requested-power capability.

Commercial Revenue-at-Risk, alert prioritization, and the typed FastAPI delivery layer are
implemented. Dashboards, monitoring, RAG, and agents are not implemented yet.

## Provenance and scientific boundary

Battery telemetry is `SIMULATED`; ENTSO-E observations are `REAL`; engineered features and residuals are `DERIVED`; fitted-model outputs are `MODEL_PREDICTION`. Fault truth may select healthy rows for offline experimental training and annotate evaluation, but it never enters model predictors or production inference.

Expected-behavior models estimate normal behavior. Delivery-risk models estimate future telemetry-defined delivery failure, not fault identity. None declares anomalies or provides operating instructions. See [TARGET_DEFINITION.md](TARGET_DEFINITION.md), [ML_METHODOLOGY.md](ML_METHODOLOGY.md), [MODEL_CARD.md](MODEL_CARD.md), and [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md).

## Reproduction

Requires Python 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy src
```

## API

Start the service with `uvicorn api.main:app --reload`. Then inspect `/api/v1/health`,
`/api/v1/assets`, `/api/v1/assets/BESS-001/status`,
`/api/v1/assets/BESS-001/delivery-risk`, and `/api/v1/alerts`. Reads return persisted outputs and
never train models or launch commercial optimization. Missing persisted analytics return a
controlled 503. Simulation controls default off; Copilot is intentionally unavailable. See
[`docs/API.md`](docs/API.md).

## Live Demo / Dashboard

Run FastAPI in Terminal 1:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run Streamlit in Terminal 2:

```powershell
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501/). The ten-page industrial operations UI is
API-first. History-only panels can use an explicitly labeled local persisted-artifact demo mode;
dashboard startup never retrains models. See [`docs/DASHBOARD.md`](docs/DASHBOARD.md).

Generate simulator-backed expected-behavior artifacts using the documented seed and duration:

```powershell
python -m models.experiment --output artifacts/models --intervals 180
```

This writes machine-generated comparisons, trusted-local joblib artifacts, metadata, and prediction/residual Parquet files. Optional figures are available through `models.plots` after installing `.[plots]`.

Run the 14-day seed-42 delivery-risk experiment without network access:

```powershell
python -m models.delivery_risk.experiment --intervals 4032
```

Outputs are written under `artifacts/models/delivery_risk/` and `reports/delivery_risk/`. The latest simulated run selected Random Forest for all horizons. Its 6-hour test PR-AUC was 0.8368, while 12-hour performance was modest and 24-hour calibration was poor; these are not real-BESS validation results.

Run the separate 10-day seed-42 anomaly experiment:

```powershell
python -m models.anomaly.experiment
```

Outputs are isolated under `artifacts/models/anomaly/` and `reports/anomaly/`. On six high-severity simulated test faults, Isolation Forest detected 5/6 but generated 17.18 false alert events per healthy day. Engineering, residual, and vote detectors each detected 2/6 with zero false alerts/day; robust peer detected 0/6. All BESS telemetry and injected faults are simulated, and no real commercial-BESS validation has occurred. Prompt 5's 24-hour probability remains poorly calibrated out of time and is descriptive context only.

Run the deterministic availability experiment:

```powershell
python -m availability.experiment
```

Availability is not one generic percentage. BESSPulse separately reports component technical state, charge/discharge MW, deliverable discharge MWh, charge headroom, and capability relative to the current request. The latest simulated run produced mean technical availability 0.99922, mean discharge power availability 0.99484, and mean discharge energy availability 0.45448. All modest requests were capability-supportable, while observed delivery still failed on 3.75% of active intervals. Definitions are in [AVAILABILITY_DEFINITIONS.md](AVAILABILITY_DEFINITIONS.md); machine results are under `reports/availability/`.

Generate causal features from already persisted data with `python -m features.build --start 2026-01-01 --end 2026-01-08 --asset-id BESS-001`. Neither command requires live ENTSO-E access. Configuration defaults are in `configs/default.yaml`; secrets such as `ENTSOE_API_TOKEN` belong only in the environment.

## Limitations

This is a short-horizon, single-asset synthetic research baseline, not a calibrated electrochemical or production BESS model. It omits cell balancing, voltage dynamics, grid-control transients, auxiliary loads, reactive power, and real-site validation. Fault magnitudes are transparent synthetic assumptions.
## Price forecasting

The compact Prompt 8 module forecasts DE-LU day-ahead prices for a timestamp-defined next 24
hours using exact 24h/168h lags, market-local calendar fields, rolling price history, transparent
baselines, and a deterministic LightGBM candidate. Selection is chronological and validation-only;
walk-forward results, negative/high-price behavior, quantiles, artifacts, and provenance are
reported under `reports/price_forecast/` and `artifacts/models/price/`. See
[`docs/PRICE_FORECASTING.md`](docs/PRICE_FORECASTING.md). Market forecasting remains supporting
context for battery reliability and future commercial-impact work.
## Revenue at Risk

BESSPulse now converts deterministic capability loss into a healthy-versus-current optimized
market-value difference using CVXPY and HiGHS. It is a counterfactual benchmark, **not actual
commercial P&L**. Historical perfect-hindsight and forecast-price modes remain separate. See
[`REVENUE_AT_RISK.md`](REVENUE_AT_RISK.md).
## Alerts

Alerts combine delivery-risk predictions, anomaly evidence, deterministic MW/MWh impact, and
counterfactual commercial context through a bounded, documented formula. They are transparent
decision support—not a black-box ranking model. The 24h risk horizon is deliberately downweighted
because of poor out-of-time calibration. See [`ALERT_INTERPRETATION.md`](ALERT_INTERPRETATION.md).
