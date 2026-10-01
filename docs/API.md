# BESSPulse API

Prompt 11 exposes persisted BESSPulse outputs through FastAPI at `/api/v1`. Interactive OpenAPI
documentation is at `/docs`; the machine schema is `/openapi.json`. The delivery layer validates
requests, resolves request-scoped services, maps controlled errors, and constructs typed responses.
Battery physics, feature engineering, model training, anomaly scoring, availability calculations,
CVXPY optimization, and alert-priority calculations remain outside route handlers.

## Local start

```powershell
python -m pip install -e ".[dev]"
uvicorn api.main:app --reload
```

Configuration includes `DATABASE_URL`, `API_HOST`, `API_PORT`, `API_LOG_LEVEL`,
`API_CORS_ORIGINS`, `ENABLE_SIMULATION_CONTROL_API`, `API_MAX_PAGE_SIZE`,
`API_DEFAULT_PAGE_SIZE`, `API_STALE_AFTER_SECONDS`, and `API_MAX_QUERY_DAYS`. CORS defaults only to
local Streamlit origins, never a credentialed wildcard.

## Routes

| Method and path | Purpose |
|---|---|
| `GET /api/v1/health` | Combined health; optional Copilot absence does not fail readiness |
| `GET /api/v1/health/live` | Process liveness |
| `GET /api/v1/health/ready` | Core dependency readiness |
| `GET /api/v1/assets` | Configured assets |
| `GET /api/v1/assets/{asset_id}/status` | Aggregation of latest persisted outputs |
| `GET /api/v1/assets/{asset_id}/availability` | Separate technical, directional MW/MWh, and request capability |
| `GET /api/v1/assets/{asset_id}/delivery-risk` | Persisted 6h/12h/24h probabilities and model versions |
| `GET /api/v1/assets/{asset_id}/revenue-risk` | Latest persisted counterfactual benchmark; never solves on GET |
| `GET /api/v1/racks/{rack_id}` | Latest observable rack state, peer context, and anomaly summary |
| `GET /api/v1/racks/{rack_id}/anomalies` | Filtered, paginated anomaly events |
| `GET /api/v1/alerts` | Filtered, paginated decision-support alerts |
| `GET /api/v1/market/latest` | Separate observed, forecast, and derived market sections |
| `POST /api/v1/simulation/fault` | Development/demo fault scheduling when enabled |
| `POST /api/v1/simulation/reset` | Safe in-memory simulation-control reset when enabled |
| `POST /api/v1/copilot/query` | Reserved typed contract; returns 503 in Prompt 11 |

Anomaly filters are `start`, `end`, `limit`, `offset`, `detector`, and `active_only`. Alert filters
are `asset_id`, `component_id`, `status`, `priority_level`, `alert_type`, `start`, `end`, `limit`,
and `offset`. Ranges require timezone-aware timestamps, `start < end`, and no more than the
configured maximum. The current route default is 50 and the configured maximum defaults to 200.

## Response semantics

Every operational timestamp is ISO-8601 UTC. Latest-state responses include `as_of_utc`,
`age_seconds`, and `is_stale`; default staleness is 900 seconds. Old data is explicitly marked.

Provenance identifies type, source, source version, generation time, and optional model and feature
versions. `REAL`, `SIMULATED`, `DERIVED`, `MODEL_PREDICTION`, `COUNTERFACTUAL`, and
`DECISION_SUPPORT` remain distinct. Observed ENTSO-E fields and forecasts occupy different market
sections. Anomaly scores are not probabilities. Operational endpoints do not serialize fault
labels, future targets, time-to-failure truth, or evaluation-only severity.

The 24-hour delivery-risk calibration entry is always `limited` with an out-of-time warning.
Revenue-risk responses always include “Counterfactual historical simulation, not actual commercial
P&L.” Attribution fields are non-causal restoration impacts.

## Errors and request tracing

Errors use `error_code`, `message`, optional `details`, `request_id`, and `timestamp_utc` with real
HTTP status codes. Responses carry `X-Request-ID`; a bounded caller-supplied value is honored.
Logging records request ID, method, path, status, and duration but not bodies, authorization
headers, tokens, database URLs, exception messages, or stack traces.

## Controls and security boundary

Simulation controls default off and accept only the canonical fault enum plus known components.
Reset cannot delete database content. These routes have a future authorization boundary but no
authentication in Prompt 11 and must not be exposed publicly. Copilot has no Gemini/OpenAI coupling
and returns `503 AI Copilot is not configured.` Core startup needs no LLM credentials.

There is no training, retraining, raw-database, model-object, or general debug endpoint. The API is
not yet internet-hardened: authentication, authorization, TLS termination, external rate limiting,
deployment packaging, and production observability remain later work.
