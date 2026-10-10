# BESSPulse final acceptance matrix

Generated: 2026-10-10T12:46:07.804019Z. `Live-tested` means exercised against a running process or external/container service during this pass. `NOT AVAILABLE` is never counted as a pass.

| Subsystem | Implemented | Tested | Live-tested | Data source | Status | Known limitation |
|---|---|---|---|---|---|---|
| Simulator and fault library | Yes | Yes | Yes | SIMULATED | PASS | Simulator realism does not prove real-site behavior. |
| ENTSO-E ingestion | Yes | Yes | No | REAL | NOT AVAILABLE | No `ENTSOE_API_TOKEN` in the acceptance process; stored REAL price data was used instead. |
| Stored DE-LU market data | Yes | Yes | Yes, API/UI/pipeline | REAL | PASS WITH LIMITATION | Only day-ahead price is populated; load/wind/solar actuals and forecasts are absent. Coverage ends 2025-01-01. |
| Battery/market features | Yes | Yes | Yes | SIMULATED + REAL + DERIVED | PASS WITH LIMITATION | Fields requiring absent load/wind/solar remain null; no data was invented. |
| Expected Power | Yes | Yes | Yes | SIMULATED / MODEL_PREDICTION | PASS WITH LIMITATION | Real-BESS generalization is unproven. |
| Expected Temperature | Yes | Yes | Yes | SIMULATED / MODEL_PREDICTION | PASS WITH LIMITATION | Real-BESS generalization is unproven. |
| Delivery Risk 6h | Yes | Yes | Yes | SIMULATED / MODEL_PREDICTION | PASS WITH LIMITATION | Simulator-only validation. |
| Delivery Risk 12h | Yes | Yes | Yes | SIMULATED / MODEL_PREDICTION | PASS WITH LIMITATION | Low held-out recall (0.2366). |
| Delivery Risk 24h | Yes | Yes | Yes | SIMULATED / MODEL_PREDICTION | PASS WITH LIMITATION | Severe out-of-time calibration shift; every held-out row classified positive. |
| Anomaly Detection | Yes | Yes | Yes | SIMULATED / DERIVED | PASS WITH LIMITATION | Isolation Forest has 57.83 false alerts/day; several detectors miss most events. |
| Availability | Yes | Yes | Yes | DERIVED from SIMULATED telemetry | PASS | Capability and observed performance remain separate. |
| Price Forecast | Yes | Yes | Yes | REAL / MODEL_PREDICTION | PASS WITH LIMITATION | Simple blend beat LightGBM; distribution shift, weak tails, and uncalibrated quantiles remain. |
| Commercial / RevenueAtRisk | Yes | Yes | Yes | REAL price + SIMULATED capability + COUNTERFACTUAL | PASS WITH LIMITATION | Perfect-hindsight historical benchmark; not actual P&L. |
| Alerts | Yes | Yes | Yes | DERIVED decision support | PASS WITH LIMITATION | Prior-alert fraction was 40% on simulated evaluation. |
| FastAPI | Yes | Yes | Yes, actual HTTP | Persisted outputs | PASS | Authentication/rate limiting are not production-grade. |
| Streamlit | Yes | Yes | Yes, server + 10 pages | FastAPI | PASS | Fixed demo timestamps are stale by design. |
| Gemini Copilot | No live provider | Security contract tested | No | AI-GENERATED EXPLANATION when configured | NOT AVAILABLE | No `GEMINI_API_KEY`; operational tool layer is absent and fails closed. |
| RAG | Yes | Yes | Yes, host + container | Approved docs | PASS | Local retrieval is extractive, not Gemini synthesis. |
| MLflow | Yes | Yes | Yes, local store | Measured model outputs | PASS | Local SQLite store; no remote URI configured. |
| Monitoring | Yes | Yes | Yes | Persisted outputs/test distributions | PASS WITH LIMITATION | Historical demo is correctly stale; no live production stream. |
| SQLite demo DB | Yes | Yes | Yes | Local demo | PASS | Not a production backend. |
| PostgreSQL | Yes | Yes | Yes, PostgreSQL 16 | Container dev DB | PASS | Not the user's Supabase project. |
| Supabase | Supported | Config tested | No | External PostgreSQL | NOT AVAILABLE | No database/Supabase variables in this process. |
| Docker | Yes | Yes | Yes | Production images/compose | PASS | Empty container DB reports no data until seeded. |
| CI | Yes | Yes | Prior remote + local gates | GitHub Actions | PASS | Live integrations are opt-in. |
| Public deployment | Hook/config present | No-hook path tested | No | External | NOT AVAILABLE | No authorized target or verified public URLs. |

Overall status: **PASS WITH LIMITATIONS**. The local technical demo and research prototype are accepted. Public portfolio use requires the documented provenance labels and configured deployment. This is not accepted for commercial production, autonomous control, safety decisions, or actual P&L reporting.
