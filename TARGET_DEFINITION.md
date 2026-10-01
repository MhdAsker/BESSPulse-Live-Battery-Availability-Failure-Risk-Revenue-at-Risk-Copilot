# Requested-power delivery-risk targets

## Interval failure

An interval is eligible only when `abs(requested_power_mw) > minimum_request_threshold_mw`. The typed default is `0.1 MW`. For eligible intervals:

```text
delivery_ratio = abs(actual_power_mw) / abs(requested_power_mw)
delivery_failure = delivery_ratio < failure_delivery_ratio_threshold
```

The typed failure-ratio default is `0.95`. Charge and discharge use magnitudes, consistent with positive-discharge/negative-charge telemetry. Idle and near-zero requests cannot be failure events, although their feature rows may receive future-risk labels based on later eligible active intervals.

## Future horizons

For horizon `h`, `failure_within_h` is one when at least one telemetry-derived delivery failure occurs in `(T, T+h]`. An event exactly at `T` is excluded; one exactly at `T+h` is included. Implemented horizons are 6, 12, and 24 hours.

Fault presence is not the target. Fault identity, family, severity, and simulator ground truth are evaluation-only metadata.

## Censoring and purging

If observation ends before `T+h`, the target is unavailable (`NaN`), not zero. Horizon datasets exclude censored rows. Irregular timestamps use timestamp searches rather than row offsets.

Each horizon is split independently into 55% train, 15% validation, 15% calibration, and 15% test. From train, validation, and calibration, rows whose target window reaches the next partition are removed. Thus the preceding retained timestamp plus `h` is strictly earlier than the next partition. Known fault boundaries move to event starts when they intersect a proposed boundary.

Train fits preprocessing and classifiers; validation selects the classifier; calibration fits identity/Platt/isotonic mappings; test is used once after those choices.
