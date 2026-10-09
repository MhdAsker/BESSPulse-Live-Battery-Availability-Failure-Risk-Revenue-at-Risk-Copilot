# BESSPulse API

FastAPI exposes persisted operational and analytical outputs under `/api/v1`; interactive OpenAPI documentation is at `/docs`. Route handlers do not train models, score raw datasets, optimize dispatch, reindex RAG, or download market history.

## Runtime configuration

Development defaults to SQLite. `APP_ENV=production` requires a PostgreSQL `DATABASE_URL`, rejects wildcard CORS, and rejects enabled simulation controls. `API_CORS_ORIGINS` is a comma-separated allowlist. Production logs are JSON and exclude request bodies, authorization headers, URLs, and exception messages.

## Routes

| Route | Purpose |
|---|---|
| `GET /api/v1/health` | Combined service status |
| `GET /api/v1/health/live` | Process-only liveness |
| `GET /api/v1/health/ready` | Database-backed readiness |
| `GET /api/v1/system/info` | Non-secret build/backend/model metadata |
| `GET /api/v1/assets` | Configured assets |
| `GET /api/v1/assets/{asset_id}/status` | Latest persisted asset state |
| `GET /api/v1/assets/{asset_id}/availability` | Technical, directional MW/MWh, and requested-power capability |
| `GET /api/v1/assets/{asset_id}/delivery-risk` | Persisted 6/12/24-hour model probabilities |
| `GET /api/v1/assets/{asset_id}/revenue-risk` | Latest counterfactual benchmark; never solves on GET |
| `GET /api/v1/racks/{rack_id}` | Rack state, peer context, and anomaly summary |
| `GET /api/v1/racks/{rack_id}/anomalies` | Filtered anomaly events |
| `GET /api/v1/alerts` | Filtered operational decision-support alerts |
| `GET /api/v1/market/latest` | Separate observed, forecast, and derived market fields |
| `GET /api/v1/monitoring` | Persisted model/data health with filters |
| `POST /api/v1/copilot/query` | Optional grounded documentation retrieval |

Simulation fault/reset routes exist only outside production and still require `ENABLE_SIMULATION_CONTROL_API=true`.

## Response and security semantics

Operational timestamps are UTC and latest-state responses expose freshness. Provenance types remain distinct. Anomaly scores are not probabilities. The 24-hour delivery-risk entry remains calibration-limited. Revenue-at-Risk always includes its counterfactual disclaimer. No endpoint serializes fault labels, future targets, time-to-failure truth, environment variables, arbitrary SQL, model objects, or filesystem paths.

Errors contain a controlled code/message, request ID, and UTC timestamp without stack traces. Responses carry `X-Request-ID`; logs record method, path, status, and duration. Authentication is not implemented, so the supported public configuration is read-only demo access behind provider HTTPS and external rate protection.
