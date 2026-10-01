# ENTSO-E integration

## Purpose and supported data

ENTSO-E supplies real German electricity-market and power-system context for later comparison with simulated BESS telemetry. The integration supports day-ahead prices, actual total load, total load forecast, actual wind-onshore, wind-offshore and solar generation, and corresponding wind/solar forecasts when ENTSO-E publishes them. Cross-border flows are out of scope.

## Region mapping

The public region name is `DE-LU`. It maps to ENTSO-E domain `10Y1001A1001A82H` and market timezone `Europe/Berlin`. These values and the API endpoint live in typed `MarketConfig` and `configs/default.yaml`, rather than query code.

## Credential configuration

Set the token in the process environment only:

```powershell
$env:ENTSOE_API_TOKEN = '<your token>'
```

The CLI has no token option. `.env.example` contains an empty variable only. Request errors, logs, filenames, metadata, summaries, and database rows never include the credential. Token-like query parameters are explicitly redacted from HTTP-library diagnostic logs.

## Time and interval policy

Python client boundaries must be timezone-aware and use internal `[start, end)` semantics. They are converted using timezone rules to UTC before query construction; naive datetimes are rejected. Date-only CLI values are explicitly interpreted as midnight in `Europe/Berlin`, so spring days can contain 23 hours and autumn days 25. Every normalized timestamp is UTC. ENTSO-E period positions are reconstructed as `start + (position - 1) * resolution`.

The parser currently supports `PT15M`, `PT30M`, `PT60M`, and other positive ISO durations composed only of hours/minutes. Unsupported forms fail rather than being guessed. The client sends one interval request and does not yet chunk long ranges; use modest ranges that comply with ENTSO-E constraints.

## Queries

| Dataset | ENTSO-E document/process | Domain parameter |
|---|---|---|
| Day-ahead prices | `A44` | `in_Domain` and `out_Domain` |
| Actual load | `A65` / `A16` | `outBiddingZone_Domain` |
| Load forecast | `A65` / `A01` | `outBiddingZone_Domain` |
| Actual generation by type | `A75` / `A16` | `in_Domain` |
| Wind/solar forecast by type | `A69` / `A01` | `in_Domain` |

Production types `B16`, `B18`, and `B19` map to solar, wind offshore, and wind onshore. Component observations remain `REAL`. Wind is aggregated only when onshore and offshore exist at the same timestamp, and that output is labeled `DERIVED`. Missing components are never treated as zero.

## Raw-data strategy

Successful HTTP response bytes are stored unchanged under `data/raw/entsoe/` before parsing. Filenames contain dataset, safe region, UTC request interval, and the first 16 characters of the SHA-256 hash. Adjacent JSON records source, interval, retrieval time, status, full hash, and `REAL` provenance. Identical content is deduplicated by hash and never overwritten. Normalized deduplication never deletes raw artifacts.

## Normalized schema and duplicate policy

`market_data` is a long-form table containing UTC timestamp, region, metric, numeric value, unit, actual/forecast type, source, provenance, retrieval time, raw hash, document/series identifiers, ENTSO-E classifications, resolution, and creation time. The principal query index is `(market_region, metric, timestamp_utc)`.

Normalized source identity is `(market_region, metric, timestamp_utc, series_type, timeseries_mrid, raw_content_hash)`. Re-ingesting identical source content is idempotent. A changed raw document is retained as a distinguishable revision. Exact duplicates inside one document collapse deterministically; conflicting values with the same series/metric/time identity raise validation errors. Legitimate distinct time-series IDs remain separate.

## Missing data and validation

Ingestion never interpolates, fills, clips, or replaces missing observations with zero. Negative prices are retained. `validate_series_continuity` reports expected interval/count, actual count, missing timestamps, and duplicate timestamps for one logical series. Warnings are returned in the ingestion summary. Optional forecasts with no matching ENTSO-E data are reported as unavailable, not successful or fabricated.

## Retries and failures

Network failures, HTTP 429, and HTTP 5xx receive at most two retries after the first attempt. Backoff is exponential; 429 honors numeric `Retry-After` up to 60 seconds. Authentication and other permanent 4xx failures fail immediately. Malformed XML, unsupported production types/resolutions, rate-limit exhaustion, request failures, and no-data documents have controlled exception types.

## Manual use

Validate an interval without credentials or a network request:

```powershell
python -m data.entsoe --start 2026-01-01 --end 2026-01-03 --dry-run
```

Perform ingestion after setting `ENTSOE_API_TOKEN`:

```powershell
python -m data.entsoe --start 2025-01-01 --end 2025-01-03
```

Date-only end is exclusive. By default normalized rows use `DATABASE_URL` when set, otherwise `sqlite:///data/besspulse.db`.

The optional live test is doubly gated:

```powershell
$env:RUN_ENTSOE_INTEGRATION_TESTS = '1'
python -m pytest tests/test_entsoe_live.py -v
```

## Known limitations

The client does not yet chunk long queries or model publication/version timestamps as a separate revision dimension. Forecast availability varies by period and ENTSO-E publication practice. ENTSO-E may revise classifications or return overlapping series; unsupported production types intentionally require an explicit mapping decision. SQLite can lose timezone-display metadata even though inserted instants are normalized to UTC; PostgreSQL is the intended later production store.

