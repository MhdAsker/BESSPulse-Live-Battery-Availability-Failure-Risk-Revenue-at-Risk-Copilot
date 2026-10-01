# Availability engine

The engine is a database-independent deterministic service under `src/availability`. It calculates rack capability, conservatively aggregates through PCS converter limits, creates a site snapshot, assesses the active request, and derives elapsed-time-weighted history and operational events.

```text
rack telemetry -> rack technical/MW/MWh capability
              -> PCS rack aggregation + converter constraint
              -> site directional technical/MW/MWh capability
              -> requested-power and energy sufficiency
              -> time-weighted summaries, downtime, derating events
```

The calculation does not import or require expected-behavior models, delivery-risk models, or anomaly detectors. Prompt 5 probabilities and Prompt 6 events are joined only by the experiment after physical snapshots exist. Fault truth is likewise used only in a post-calculation diagnostic report.

Run a short no-fault development calculation:

```powershell
python -m availability.engine --asset-id BESS-001 --start 2026-01-01T00:00:00Z --end 2026-01-01T06:00:00Z --output reports/availability/development.parquet
```

Run the reproducible 10-day Prompt 7 experiment:

```powershell
python -m availability.experiment
```

The experiment uses the same seed, dates, requests, ambient profile, and injected schedule as Prompt 6, but only observable telemetry enters availability formulas. Reports include site snapshots, rack/PCS capability, component downtime, deterministic downtime/derating events, fault-truth diagnostics, and descriptive-only risk/anomaly context.

Persistence uses `availability_snapshots` with unique `(timestamp_utc, asset_id)`. Reprocessing skips existing identities instead of duplicating them.

Known limitations include the reduced-order simulator, an SOH proxy rather than measured electrochemical SOH, no reactive-power capability, no cell-level constraints, exact-timestamp history rather than live stream reconciliation, and a PCS derating factor that must be normalized against rack capability because simulator telemetry reports it for the current request direction.
