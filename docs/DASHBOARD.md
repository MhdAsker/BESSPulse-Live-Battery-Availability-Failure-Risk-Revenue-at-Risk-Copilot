# BESSPulse Streamlit dashboard

The ten-page dashboard is a server-side API client and presentation layer. It never receives database, Gemini, ENTSO-E, or Supabase server credentials. Set `BESSPULSE_API_URL` to the deployed HTTPS API base path.

## Pages

1. Fleet Overview: nameplate, availability, SOC/RTE, delivery risk, alerts, and Revenue-at-Risk.
2. Live Asset: animated battery/power direction and persisted telemetry.
3. Rack Heatmap: PCS/rack layout and rack detail.
4. ML Health: saved expected-behavior, risk, and anomaly evaluations.
5. Delivery Risk: horizon probabilities, versions, calibration, and associations.
6. Market Context: real observed DE-LU data separated from forecasts.
7. Revenue-at-Risk: healthy/current counterfactuals and restoration attribution.
8. Alerts: evidence, MW/MWh impact, commercial context, and transparent formula.
9. AI Copilot: approved-document retrieval, citations, and tool traces; unavailable capabilities fail honestly.
10. Model Monitoring: persisted freshness, drift, quality, prediction, performance, and artifact observations.

Streamlit's automatic multipage sidebar is disabled so only the product navigation renders. Auto-refresh and UTC/Berlin display persist in session state. The narrow layout remains usable through wrapping cards and Streamlit's responsive columns, though the experience is desktop-first.

## Data behavior

The normal direction is domain/database → FastAPI → Streamlit. A clearly labeled local artifact fallback supports development demonstrations for history-heavy visualizations. Disable it in production with `DASHBOARD_DEMO_FALLBACK=false`.

Every public view preserves the scientific labels: simulated BESS telemetry, real ENTSO-E observations, derived engineering analytics, model predictions, and counterfactual commercial outputs. The 24-hour calibration limitation and Revenue-at-Risk disclaimer remain visible. Missing optional services render explicit unavailable/no-data states rather than fabricated values.

The dashboard is unauthenticated. Deploy it only as a read-only portfolio/demo until identity, authorization, and tenant isolation are implemented.
