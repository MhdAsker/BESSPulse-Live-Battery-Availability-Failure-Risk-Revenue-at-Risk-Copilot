# Experiment tracking

`ExperimentTracker` uses `MLFLOW_TRACKING_URI`, defaulting to the local SQLite store `sqlite:///mlruns.db`. Experiment names are stable by model family: expected power, expected temperature, anomaly detection, delivery risk, and price forecasting. Parameters, metrics, tags, and trusted artifacts can be logged through one adapter.

Keys resembling secrets, tokens, passwords, credentials, API keys, authorization, or database URLs are discarded before parameter/tag logging. Remote tracking failures are fail-open by default so model training is not made unavailable by telemetry. Set `fail_open=False` for validation jobs that require tracking.

Delivery-risk runs receive a tag for the known 24-hour calibration limitation. Registration requires `validated=True`; this creates only a candidate and never promotes it automatically. CI tests use a temporary local MLflow store and require no external server.
