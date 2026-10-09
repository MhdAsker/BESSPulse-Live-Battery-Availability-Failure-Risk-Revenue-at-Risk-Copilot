# Deployment

## Service split

BESSPulse deploys as two independent non-root containers:

- `besspulse-api`: FastAPI on provider `$PORT`, with PostgreSQL through `DATABASE_URL`.
- `besspulse-dashboard`: Streamlit on provider `$PORT`, with only `BESSPULSE_API_URL`.

Training, ENTSO-E ingestion, RAG indexing, monitoring, and other periodic work are separate jobs. Web startup never retrains or downloads data.

## Render Blueprint

[`render.yaml`](../render.yaml) defines both web services. In Render:

1. Create a Blueprint from this repository and branch.
2. For the API, set `DATABASE_URL` to the Supabase PostgreSQL connection string, preferably the transaction pooler URI with its required SSL query parameters.
3. Set `API_CORS_ORIGINS` to the final HTTPS dashboard origin. Do not use `*`.
4. Keep `ENABLE_SIMULATION_CONTROL_API=false`.
5. Add `ENTSOE_API_TOKEN` and `GEMINI_API_KEY` only if those optional features are enabled.
6. Set `BUILD_COMMIT` to the deployed Git SHA when the platform does not inject it automatically.
7. For the dashboard, set `BESSPULSE_API_URL=https://<api-host>/api/v1`.
8. Run `alembic upgrade head` as the API pre-deploy/release command before routing traffic.

The deployment configuration intentionally does not contain credentials. Render Blueprint values marked `sync: false` must be entered in the platform secret store. Render health checks use `/api/v1/health/live` for FastAPI and `/_stcore/health` for Streamlit.

## Streamlit Community Cloud alternative

Use `dashboard/app.py` as the entry point. Configure only the API URL and presentation settings in the application settings. Backend database, Gemini, ENTSO-E, and Supabase secret credentials must not be placed in Streamlit configuration. The API URL must use HTTPS to avoid mixed-content failures.

## Database migration

Production schema changes are explicit:

```text
alembic upgrade head
```

Do not call `Base.metadata.create_all()` from web workers. Do not run migrations concurrently from every replica. Verify the current revision with `alembic current` and schema drift with `alembic check` against a safe database.

## CORS and public mode

Production startup rejects SQLite, wildcard CORS, and enabled simulation controls. The public dashboard is read-only and unauthenticated; it is appropriate only for a portfolio/demo. Add authentication and authorization before exposing operator mutations or tenant data.

## Artifacts and RAG

Generated model binaries are not committed or copied into images. The API can serve persisted predictions without them and reports artifact availability separately. A real inference deployment must provide verified artifacts through a read-only volume, object store, or governed MLflow registry.

The local RAG backend requires an explicit `python -m rag.index` job and persistent `.rag` storage. The API never rebuilds the index at startup. `RAG_ENABLED` should remain false until an index is provisioned. pgvector is not currently implemented and must not be selected.

## Deployment verification

After deployment, verify actual HTTPS URLs:

```text
GET <api>/api/v1/health/live
GET <api>/api/v1/health/ready
GET <api>/api/v1/system/info
GET <api>/api/v1/assets
GET <api>/api/v1/monitoring
GET <dashboard>/_stcore/health
```

Then verify all ten dashboard pages, provenance labels, the 24-hour calibration warning, the Revenue-at-Risk disclaimer, CORS from the dashboard origin, and rejection from an unapproved origin. Do not claim a public deployment until these checks pass.

## GitHub deployment hooks

The deployment workflow requires repository secrets named `RENDER_API_DEPLOY_HOOK` and `RENDER_DASHBOARD_DEPLOY_HOOK`. It runs automatically only after successful CI on `main`, or manually through `workflow_dispatch`. Hook values are never printed.
