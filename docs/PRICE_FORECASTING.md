# Price forecasting

## Purpose and scope

BESSPulse forecasts normalized DE-LU day-ahead prices only to supply forward-looking market
context for later BESS commercial analysis. It is intentionally a compact, transparent model,
not a trading system, price-research benchmark, or financial guarantee. Dispatch optimization,
revenue-at-risk, attribution, alerts, APIs, and dashboards remain out of scope for Prompt 8.

The target is `day_ahead_price_eur_per_mwh`, sourced from REAL ENTSO-E observations. Every
forecast is marked `MODEL_PREDICTION`. Battery telemetry, availability, faults, anomalies, and
delivery-risk outputs are prohibited model inputs.

## Time and horizon semantics

Delivery timestamps and forecast origins are stored in UTC. Calendar fields are converted to
`Europe/Berlin`. `hour_of_week = day_of_week * 24 + hour_of_day`; the autumn repeated hour has
the same hour-of-week but a distinct `utc_offset_hours`, while the spring skipped hour simply has
no delivery timestamp. No 24-row-day assumption is made.

The default horizon is the timestamp interval `(origin, origin + 24h]`. The number of rows is
derived from the observed resolution: 96 at 15 minutes or 24 hourly. Each output records its
origin, target timestamp, and `horizon_minutes`.

## Feature availability contract

| Feature | Source | Timestamp meaning | Available by origin | Legal for target T |
|---|---|---|---|---|
| `price_lag_24h` | REAL ENTSO-E price | delivery at T-24h | yes, exact timestamp only | yes |
| `price_lag_168h` | REAL ENTSO-E price | delivery at T-168h | yes, exact timestamp only | yes |
| rolling price mean/volatility | DERIVED FROM REAL | observations in `(T-24h-window,T-24h]` | yes for every target through O+24h | yes |
| calendar and UTC offset | DERIVED | target delivery timestamp | deterministic | yes |
| load/wind/solar forecast | REAL ENTSO-E forecast | target delivery timestamp | only with known causal forecast vintage | conditionally |
| future actual load/generation | ex-post actual | target delivery timestamp | no | prohibited |

Exact lags use timestamp lookup; a missing timestamp returns `NaN`, never a nearby or future
observation. Required lag-missing rows are unavailable to training or inference. For a 24-hour
forecast contract, rolling windows end at T-24h; this is conservative at short horizons and
prevents historical rolling-origin rows from seeing prices learned after origin. Inference filters
price history at the explicit origin and requires every target to be later than that origin.

ENTSO-E system forecasts are supported by the feature contract but excluded from the current
empirical experiment. The repository does not store publication vintages separately, so a latest
historical revision cannot yet prove what was available at an old forecast origin. Actual system
values are never substituted.

## Models and selection

Required baselines are exact lag-24, exact lag-168, and a train-only median by market-local
hour-of-week. A simple equal blend is also evaluated. The point ML candidate is a deterministic,
moderately sized `LGBMRegressor` using both lags, 24-hour/7-day price mean and volatility,
hour/day/weekend fields, hour-of-week, and UTC offset.

Train, validation, and test are chronological and never shuffled. Selection minimizes validation
MAE. If a simpler candidate is within 2% of the best MAE, the fixed simplicity order chooses it.
Only after selection is the chosen estimator refit on train+validation and evaluated once on test.
The high-price threshold is the training-only 95th percentile.

Expanding-window backtests refit at each historical origin, use only rows at or before that
origin, and evaluate targets in the next timestamp-defined 24 hours. The default report limits
this to 12 daily origins to keep runtime controlled.

## Metrics and price behavior

Reports include MAE, RMSE, R-squared, mean prediction-minus-actual bias, median absolute error,
and P95 absolute error. Negative prices and extremes are retained without clipping,
winsorization, or log transforms. Negative-price and training-threshold high-price subsets are
reported separately.

Independent LightGBM P10/P50/P90 models report pinball loss, empirical coverage, and raw crossing
fractions. Grouped output may be made monotonic by sorting each row; raw predictions remain the
evaluation basis.

## Artifacts and storage

Trusted-local joblib artifacts and JSON metadata live under `artifacts/models/price/`. Metadata
records the exact feature list, model/version, target, market, data hash, resolution, seed,
hyperparameters, selection rule, chronological intervals, provenance, and empirical metrics.
Loading joblib from untrusted sources is unsafe.

`price_predictions` uniquely identifies a forecast by origin, target, market, model version, and
quantile. Point forecasts use the internal quantile sentinel `-1.0`, avoiding SQLite's multiple
NULL uniqueness behavior. Repeated identical writes are no-ops. Reports are in
`reports/price_forecast/`.

## Current empirical experiment and limitations

The Prompt 8 run uses a live-retrieved REAL ENTSO-E response. The response contains parallel
classification sequences at PT15M and PT60M; the experiment explicitly selects the coherent
finest PT15M sequence rather than mixing resolutions at coincident timestamps. The continuity
report exposes all missing intervals. No missing target or system forecast is fabricated.

The experiment covers one historical year only, contains gaps, and uses latest-retrieved data
rather than archived publication vintages. Distribution shift is visible between validation and
test, and errors are substantially worse during negative and high-price regimes. Quantile
coverage is not calibrated. These outputs are suitable for research context only.
