# BESSPulse Streamlit Dashboard

The Prompt 12 dashboard is a desktop-first operations interface at `dashboard/app.py`. Its normal
dependency direction is domain/database → FastAPI → Streamlit. The UI validates Prompt 11 response
models through a reusable `httpx` client and never trains models, scores anomalies, calculates
availability, optimizes dispatch, or reprioritizes alerts.

## Start locally

Terminal 1:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501/). Configure the API with
`BESSPULSE_API_URL`; the default is `http://localhost:8000/api/v1`.

## Product pages

1. Fleet Overview — nameplate, directional availability, SOC/RTE, 6h/12h/24h risk, alerts, and Revenue-at-Risk.
2. Live Asset — animated battery/power direction and synchronized persisted telemetry panels.
3. Rack Heatmap — 4 PCS groups × 8 labeled racks, selectable observable/analytic metric, and rack detail.
4. ML Health — saved expected-power, expected-temperature, delivery-risk, and anomaly evaluations.
5. Delivery Risk — probability bars, model versions, Brier scores, reliability, SHAP associations, and offline outcomes.
6. Market Context — REAL DE-LU prices, separate model forecast, generation/load availability, and price metadata.
7. Revenue at Risk — healthy/current counterfactuals, dispatch, cumulative revenue, and restoration attribution.
8. Alerts — Prompt 10 filtering, evidence, impacts, confidence, source versions, and formula explanation.
9. AI Copilot — polished Prompt 13-ready shell that calls the API and shows an honest unavailable state.
10. Model Monitoring — artifact/version inventory and explicit `NOT YET ACTIVE` drift placeholders.

## Visual and interaction system

Colors are centralized in `dashboard/theme.py` and `.streamlit/config.toml`: graphite backgrounds,
steel surfaces, battery green for healthy capability, cyan for grid/market context, amber for
warnings/counterfactuals, red for critical conditions, and purple for simulated telemetry. Every
provenance color also carries a text label. CSS provides compact cards, hover elevation, status
pulse, critical alert pulse, animated SOC fill, and subtle directional power flow. Plotly figures
share transparent dark surfaces, accessible labels, units, restrained grids, and consistent hover.

Auto-refresh is selectable as Off/10s/30s/60s and defaults to 30 seconds. Time display is explicit
UTC or Europe/Berlin. Both settings persist in Streamlit session state with the selected asset,
rack, page, alert state, and Copilot conversation shell.

## API and demo data

The sidebar reports API ONLINE, DEGRADED, or OFFLINE from `/health`; it never fabricates status.
Safe GET requests use bounded retries and timeouts. Errors produce friendly categories and a
technical-detail expander without stack traces or secrets.

The Prompt 11 API currently exposes latest state but not history or a fleet rack-list endpoint.
Existing Prompt 5–10 Parquet/CSV/JSON artifacts therefore provide an explicit, optional local demo
adapter for history-heavy charts and the 32-rack layout. It is labeled `DEMO / LOCAL DATA MODE` and
never runs training or scientific computation. Disable “Allow local artifact demo fallback” in the
sidebar for API-only behavior. Regenerate existing artifacts using their documented Prompt 5–10
experiment commands; dashboard startup itself performs no training.

## Scientific boundary and limitations

- BESS telemetry is visibly `SIMULATED`; ENTSO-E observations are `REAL`.
- Model forecasts are `MODEL PREDICTION`; anomaly scores are never called probabilities.
- Availability retains technical, directional MW/MWh, and requested-power concepts.
- The 24h risk remains visible with `CALIBRATION LIMITED` and poor-reliability text.
- Revenue-at-Risk is always `COUNTERFACTUAL` with “Counterfactual historical simulation, not actual commercial P&L.”
- Historical failure markers appear only as `SIMULATED EVALUATION GROUND TRUTH`.
- No invented health score, drift metric, live fault truth, LLM answer, or causal attribution is shown.

The dashboard is not authenticated and is not ready for public internet exposure. Production auth,
deployment, monitoring, model-drift services, agents, RAG, and durable conversations remain later
phases.
