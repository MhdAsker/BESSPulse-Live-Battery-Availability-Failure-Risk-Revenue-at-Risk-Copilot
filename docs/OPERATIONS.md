# Scheduled operations

Use platform cron jobs or a dedicated worker. Do not tie periodic work to FastAPI uptime.

| Job | Command | Notes |
|---|---|---|
| ENTSO-E ingestion | `python -m data.entsoe ...` | Requires `ENTSOE_API_TOKEN`; bounded HTTP retries |
| Feature snapshot | `python -m features.build ...` | Causal, timestamp-bounded input |
| Availability snapshot | `python -m availability.engine ...` | Deterministic calculation |
| Commercial benchmark | `python -m commercial.benchmark ...` | Counterfactual, persisted output |
| Monitoring | `python -m monitoring.service --asset BESS-001 --persist` | Reads persisted timestamps and writes monitoring records |
| RAG refresh | `python -m rag.index` | Explicit content-hash incremental job |
| MLflow logging | `python -m tracking.registry ...` | Logs completed validated training outputs |

The existing expected-behavior, delivery-risk, anomaly, and price experiment commands are offline training/evaluation workflows, not production schedulers. Production inference runners for those families have not yet been packaged as independent CLIs; use persisted outputs until that deployment gap is closed.
