# Causal feature pipeline

## Architecture and contracts

`features.battery` builds separate site and rack frames. `features.peer` adds same-timestamp rack comparisons. `features.market` pivots normalized ENTSO-E long-form observations, calculates strict aggregates/lags, and aligns market rows backward to site timestamps. `features.validation` owns timestamp, identity, leakage, and infinity guards. `features.build` generates content-addressed Parquet snapshots.

Site identity is `(timestamp_utc, asset_id)` and rack identity is `(timestamp_utc, rack_id)`. Duplicate identities fail before sorting or aggregation. Input order is normalized deterministically by entity and UTC time after identity validation. Naive timestamps fail.

## Causal windows and lags

All rolling features use non-centered, right-closed trailing intervals `(t-window, t]`. Default production configuration requires the complete named elapsed duration before returning a value; initial history is `NaN`. Sample standard deviation requires at least two observations. Exposure and availability features are observation-weighted fractions, which is appropriate for regular telemetry but must be reconsidered for irregular sampling.

Price and RTE lags use exact timestamp matching, not row shifts. A 24-hour lag is `t-24h` in UTC, including across DST. Missing exact lag timestamps stay `NaN`.

## Peer strategy

Each rack is always removed before building its peer distribution. Available same-PCS racks with present measurements are the default peers. If their population is below `peer_min_population`, available site racks are used. Ground-truth fault labels never influence eligibility. Percentiles use midrank semantics. Robust z-score is `(x-median)/(1.4826*MAD)`; zero MAD returns zero only for a target equal to the median, otherwise `NaN`.

## Market alignment

Market observations are interpreted as interval-start values at their normalized timestamp. Battery/site timestamps use a backward as-of join with a finite configured tolerance. Exact matches are allowed; future observations are forbidden. A value expires at the earlier of the configured tolerance or its reported source-resolution boundary, so it cannot leak into the next unobserved market interval. This is a causal technical alignment, not a claim about historical publication/vintage availability.

Actual and forecast market series remain explicitly separate. No forecast-error feature is generated. Actual wind requires both onshore and offshore; renewable generation additionally requires solar. Missing components are never treated as zero.

## Leakage and numerical controls

The central validator rejects fault truth, ground-truth labels, `future_*`, `target_*`, and `failure_within_*`. Feature builders select observable fields, so such input columns are not propagated. No target generation occurs. Division-by-zero and zero-MAD cases return `NaN`; feature frames fail validation if infinity is present.

## Versioning and storage

Feature set version is `v1`. A snapshot directory contains separate `site_features.parquet`, `rack_features.parquet`, optional `market_features.parquet`, optional `combined_site_market_features.parquet`, and `metadata.json`. Dataset identity hashes the complete typed configuration, raw source frames, generated feature frames, requested UTC interval, and feature version. Repeating identical generation reuses the content-addressed snapshot without overwriting it.

Metadata records generation time, `[start,end)` input range, configuration/source/feature hashes, schema version, row count, feature count, and `DERIVED` provenance. Parquet is used instead of a database-wide ambiguous feature table.

## CLI

The generator reads already persisted telemetry and market data; it never calls ENTSO-E:

```powershell
python -m features.build --start 2026-01-01 --end 2026-01-08 --asset-id BESS-001
```

Date-only boundaries mean UTC midnight and end is exclusive. Output defaults to `data/processed/features/`. Missing site/rack telemetry is an explicit error.

## Limitations

Exposure fractions are observation-weighted rather than duration-integrated. Feature generation does not yet model market publication vintages, select among revised ENTSO-E documents, or build PCS-wide frames. Alarm codes remain observable categorical values for later explicit encoding. Calendar age is unavailable because commissioning time is not stored. No imputation, target labels, feature selection, or ML is implemented.
