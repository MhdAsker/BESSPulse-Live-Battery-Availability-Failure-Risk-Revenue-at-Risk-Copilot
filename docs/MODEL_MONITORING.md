# Model and data monitoring

Monitoring snapshots are typed and persisted in `monitoring_metrics`. Each observation records UTC time, subsystem, metric name/value, status and threshold, reference/current windows, sample count, optional model/version and asset, structured details, and provenance. `GET /api/v1/monitoring` supports scope, model, status, asset, and bounded-limit filters.

Project monitoring assumptions are configurable in `MonitoringConfig`: a 24-hour current window, a 30-day historical reference window, minimum sample count, and warning/critical thresholds. These are project defaults, not universal statistical standards. Reference inputs must come from training or a separately validated baseline and must not include current/future observations.

Implemented primitives cover freshness (`FRESH`, `STALE`, `MISSING`), missing/invalid/infinite fractions, PSI, Wasserstein distance, residual bias/absolute/P95 behavior, prediction distributions, label maturity by forecast horizon, Brier/precision/recall/PR-AUC, rolling 24-hour Brier, and non-executing artifact metadata/hash health. One-class AUC and immature samples return unavailable values rather than invented metrics. Drift means model/data change, not a battery fault.

The scheduler-ready entry point is `python -m monitoring.service`; without arguments it performs a non-persisting scheduler dry run, while `--persist` measures and stores database freshness for telemetry, market, risk, availability, alerts, and commercial outputs. It never fabricates model metrics without input data. The dashboard reads persisted observations and displays an explicit no-data state. The 24-hour delivery-risk calibration limitation remains visible. Model files are never loaded from untrusted paths during artifact health checks.
