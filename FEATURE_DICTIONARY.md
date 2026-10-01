# Feature dictionary — v1

All timestamps and keys are non-feature identity columns. Battery inputs are `SIMULATED`; copied ENTSO-E values are `REAL`; every calculation is `DERIVED`. All rolling windows are trailing, right-closed `(t-window, t]`, never centered, and use only timestamps `<= t`. With `require_full_windows=true` (default), named-window outputs remain `NaN` until the complete elapsed duration exists. `float` fields use `NaN` when undefined and never infinity.

The “later use” column is a candidate analytical role, not a statement that a model exists.

## Site feature frame

| Feature | Source/formula | Unit/type | Provenance | Window / causal availability | Missing behavior, caveat, later use |
|---|---|---|---|---|---|
| `rated_power_mw`, `rated_energy_mwh` | Typed nameplate configuration | MW, MWh / float | DERIVED | At `t` | Configuration assumption; normalization/KPI |
| `active_power_interval` | `abs(requested)>minimum_request_threshold` | bool | DERIVED | At `t` | Never missing; valid-delivery filter |
| `delivery_ratio` | `abs(actual)/abs(requested)` on active intervals | fraction / float | DERIVED | At `t` | `NaN` when inactive; delivery analytics |
| `power_residual_mw` | `actual-requested` | MW / float | DERIVED | At `t` | Negative means discharge under-delivery under the project sign convention; tracking |
| `absolute_power_residual_mw` | `abs(power_residual_mw)` | MW / float | DERIVED | At `t` | Missing follows source; tracking magnitude |
| `normalized_power_residual` | residual/rated power | fraction / float | DERIVED | At `t` | Nameplate-normalized; cross-asset comparison |
| `power_tracking_error_fraction` | absolute residual/absolute request | fraction / float | DERIVED | At `t` | `NaN` when inactive; tracking |
| `rolling_power_residual_mean` | trailing residual mean | MW / float | DERIVED | `power_residual_window` | Full history required by default; drift |
| `rolling_power_residual_std` | trailing sample standard deviation | MW / float | DERIVED | `power_residual_window` | Needs two points and full history; variability |
| `rolling_absolute_power_residual` | trailing mean absolute residual | MW / float | DERIVED | `power_residual_window` | Full history required; sustained tracking error |
| `requested_power_ramp_mw`, `actual_power_ramp_mw` | current minus previous value | MW/interval / float | DERIVED | Previous observation | First row `NaN`; operating dynamics |
| `soc_change` | current minus previous SOC | fraction/interval / float | DERIVED | Previous observation | First row `NaN`; state dynamics |
| `soc_change_rate` | SOC change/elapsed hours | fraction/hour / float | DERIVED | Previous observation | First/zero-elapsed row `NaN`; energy consistency |
| `soc_volatility` | trailing sample SOC standard deviation | fraction / float | DERIVED | `soc_volatility_window` | Full history/two points required; cycling behavior |
| `time_high_soc`, `time_low_soc` | trailing observation fraction beyond configured thresholds | fraction / float | DERIVED | `soc_exposure_window` | Observation-weighted; irregular sampling is a caveat; ageing proxy |
| `distance_to_soc_min`, `distance_to_soc_max` | SOC minus configured bound / bound minus SOC | fraction / float | DERIVED | At `t` | May be negative for anomalous inputs; capability context |
| `rte_change` | current minus previous RTE | fraction/interval / float | DERIVED | Previous observation | First row `NaN`; efficiency change |
| `rte_trend` | `(RTE(t)-RTE(t-window))/window_hours` with exact lag | fraction/hour / float | DERIVED | `rte_trend_window` | `NaN` without exact lag; efficiency degradation context |
| `available_rack_fraction`, `available_pcs_fraction` | observed available count/configured count | fraction / float | DERIVED | At `t` | Not clipped; operational availability precursor |
| `available_power_fraction`, `available_energy_fraction` | available/nameplate | fraction / float | DERIVED | At `t` | Not clipped; capability context |
| `technical_availability` | minimum of rack and PCS available fractions | fraction / float | DERIVED | At `t` | Precursor, not contractual availability |
| `derating_indicator` | available power fraction `< 1-epsilon` | bool | DERIVED | At `t` | Uses observables only, never fault labels; capability flag |
| `derating_frequency` | trailing mean of derating indicator | fraction / float | DERIVED | `availability_window` | Full history required; recurring limitation |
| `alarm_active` | observed alarm count `>0` | bool | DERIVED | At `t` | Operator-observable only; alert context |
| `recent_alarm_count` | trailing sum of observed alarm counts | count / float | DERIVED | `alarm_window` | Full history required; alarm burden |
| `alarm_frequency` | trailing fraction of alarm-active intervals | fraction / float | DERIVED | `alarm_window` | Observation-weighted; alert persistence |
| `time_since_last_alarm_minutes` | elapsed time since most recent alarm at/before `t` | minutes / float | DERIVED | All prior history | `NaN` before first alarm; recency |
| `is_charging`, `is_discharging`, `is_idle` | explicit equality to operating mode | bool | DERIVED | At `t` | No ordinal encoding; operating context |

Site frame also retains observable `requested_power_mw`, `actual_power_mw`, `available_power_mw`, `available_energy_mwh`, `soc`, `rte`, component counts, mode, and alarm count for auditability. These originate from `SIMULATED` telemetry; the generated snapshot is a `DERIVED` artifact.

## Rack feature frame

| Feature | Source/formula | Unit/type | Provenance | Window / causal availability | Missing behavior, caveat, later use |
|---|---|---|---|---|---|
| `rack_power_residual_mw`, `absolute_rack_power_residual_mw` | actual-requested and its absolute value | MW / float | DERIVED | At `t` | Missing follows source; rack tracking |
| `soc_change`, `soc_change_rate`, `soc_volatility` | Same causal definitions as site | fraction, fraction/hour / float | DERIVED | Previous / `soc_volatility_window` | First/insufficient history `NaN`; rack dynamics |
| `time_high_soc`, `time_low_soc` | Trailing threshold-observation fractions | fraction / float | DERIVED | `soc_exposure_window` | Full history required; ageing proxies |
| `distance_to_soc_min`, `distance_to_soc_max` | Distance from configured bounds | fraction / float | DERIVED | At `t` | Missing follows SOC; capability |
| `recent_charge_fraction`, `recent_discharge_fraction` | Trailing fraction with actual power `<0` / `>0` | fraction / float | DERIVED | `soc_exposure_window` | Idle is in neither; operating duty |
| `temperature_vs_ambient_c` | rack mean minus ambient | °C / float | DERIVED | At `t` | `NaN` when ambient is unavailable; thermal context |
| `temperature_ramp_c` | current minus previous rack mean | °C/interval / float | DERIVED | Previous observation | First row `NaN`; rapid heating |
| `rolling_temperature_mean`, `rolling_temperature_std` | trailing mean/sample standard deviation | °C / float | DERIVED | `temperature_window` | Full history; thermal state/variability |
| `thermal_exposure_above_threshold` | trailing fraction above configured threshold | fraction / float | DERIVED | `thermal_exposure_window` | Observation-weighted; thermal stress |
| `temperature_exposure` | alias of thermal exposure above threshold | fraction / float | DERIVED | Same | Explicit ageing-feature name |
| `voltage_spread_change` | current minus previous voltage spread | V/interval / float | DERIVED | Previous observation | First row `NaN`; imbalance dynamics |
| `current_change` | current minus previous current | A/interval / float | DERIVED | Previous observation | First row `NaN`; electrical dynamics |
| `rack_power_change` | current minus previous actual rack power | MW/interval / float | DERIVED | Previous observation | First row `NaN`; dynamics |
| `rte_change`, `rte_trend` | previous change and exact-window slope | fraction/interval, fraction/hour / float | DERIVED | Previous / `rte_trend_window` | Exact lag required for trend; efficiency |
| `availability_rate` | trailing mean of observable availability | fraction / float | DERIVED | `availability_window` | Full history; operational availability |
| `alarm_indicator`, `alarm_frequency`, `time_since_last_alarm_minutes` | observable alarm-code presence and causal history | bool, fraction, minutes | DERIVED | At `t` / `alarm_window` / prior history | No injected-fault family; alerts |
| `throughput_change_mwh` | current minus previous cumulative throughput | MWh/interval / float | DERIVED | Previous observation | First row `NaN`; ageing/duty |
| `high_soc_exposure`, `low_soc_exposure` | aliases of configured SOC exposure fractions | fraction / float | DERIVED | `soc_exposure_window` | Full history; ageing context |
| `is_charging`, `is_discharging`, `is_idle` | operating-state one-hot flags | bool | DERIVED | At `t` | No ordinal encoding; operating context |

Rack frames retain observable simulated values: SOC, SOH proxy, voltage, current, temperature/spreads, powers, throughput/EFC, RTE, availability, alarm code, and operating state. `calendar_age_days` is not implemented because commissioning time is unavailable.

## Leave-one-out peer features

For each prefix in `temperature`, `voltage_spread`, `rte`, `soc`, and `power_tracking`, the following explicitly named features exist:

- `{prefix}_peer_median`: leave-one-out peer median, source units.
- `{prefix}_peer_deviation`: target minus peer median, source units.
- `{prefix}_peer_robust_zscore`: deviation divided by `1.4826 * peer_MAD`, dimensionless.
- `{prefix}_peer_percentile`: midrank empirical percentile `(less + 0.5*equal)/peer_count`, fraction.

All are `float`, `DERIVED`, and available at `t` from same-timestamp observable peers only. The target is excluded from every statistic. If peers are below `peer_min_population` or the target/measurement is missing, the metric is `NaN`. With MAD below epsilon, z-score is `0` only when the target equals the median; otherwise it is `NaN`.

`peer_count` (count/int), `peer_valid_fraction` (available candidates/all candidates, fraction/float), and `peer_group` (`PCS`, `SITE`, or missing/string) document context. Same-PCS available peers are preferred; site fallback is used only when PCS population is insufficient. Ground truth never controls eligibility.

## Market feature frame

| Feature | Source/formula | Unit/type | Provenance | Window / causal availability | Missing behavior, caveat, later use |
|---|---|---|---|---|---|
| `price_eur_per_mwh` | ENTSO-E `day_ahead_price` copied unchanged | EUR/MWh / float | REAL column in DERIVED frame | At delivery timestamp | Negative values retained; market context |
| `load_mw`, `load_forecast_mw` | ENTSO-E actual/forecast metrics | MW / float | REAL column | At timestamp | Actual and forecast stay distinct |
| `wind_onshore_generation_mw`, `wind_offshore_generation_mw`, `solar_generation_mw` | ENTSO-E actual components | MW / float | REAL columns | At timestamp | Missing remains `NaN` |
| `wind_onshore_forecast_mw`, `wind_offshore_forecast_mw`, `solar_forecast_mw` | ENTSO-E forecast components | MW / float | REAL columns | At timestamp | Never merged ambiguously with actuals |
| `wind_generation_mw` | strict onshore+offshore actual sum | MW / float | DERIVED | At `t` | `NaN` if either component missing |
| `renewable_generation_mw` | strict wind+solar actual sum | MW / float | DERIVED | At `t` | `NaN` if any required component missing |
| `residual_load_mw` | actual load-renewable generation | MW / float | DERIVED | At `t` | `NaN` if inputs missing; no actual/forecast mixing |
| `renewable_share` | renewable generation/nonzero actual load | fraction / float | DERIVED | At `t` | `NaN` for zero/missing load; not clipped above 1 |
| `wind_forecast_mw`, `renewable_forecast_mw` | strict forecast component sums | MW / float | DERIVED | At `t` | `NaN` on missing components; no forecast error feature |
| `price_lag_24h`, `price_lag_168h` | strict exact UTC timestamp lookup | EUR/MWh / float | DERIVED | `t-24h`, `t-168h` | `NaN` without exact observation; no row-count assumption |
| `rolling_price_mean` | trailing price mean | EUR/MWh / float | DERIVED | `market_price_window` | Full window required by default; context |
| `price_volatility` | trailing sample price standard deviation | EUR/MWh / float | DERIVED | `market_volatility_window` | Two points/full window required; volatility |
| `price_change_1h`, `price_change_24h` | price minus strict exact lag | EUR/MWh / float | DERIVED | 1h / 24h | `NaN` without exact lag; momentum context |
| `hour_of_day`, `day_of_week`, `hour_of_week`, `is_weekend` | UTC timestamp converted via `Europe/Berlin` rules | integer/bool | DERIVED | At `t` | DST-aware; calendar context |
| `market_source_resolution_minutes` | minimum reported source resolution present at timestamp | minutes / float | DERIVED metadata | At `t` | `NaN` without resolution; freshness interpretation |

## Combined alignment features

| Feature | Formula | Unit/type | Provenance | Availability and missing behavior |
|---|---|---|---|---|
| `market_timestamp_utc` | timestamp of backward-selected market row | UTC datetime | DERIVED lineage | Missing when no row within tolerance |
| `market_data_age_minutes` | battery timestamp-market timestamp | minutes / float | DERIVED | Nonnegative or `NaN`; future matches are forbidden |
| `market_data_available` | aligned market timestamp present | bool | DERIVED | False when stale/missing |

The combined frame is `DERIVED`. Backward as-of alignment uses `market_alignment_tolerance`; it never chooses the nearest future row and never forward-fills indefinitely.

## Delivery-risk classifier features and targets

The persisted classifier manifest records every curated candidate's family, unit, `DERIVED` provenance, causal status, inclusion flag, and inclusion/exclusion reason. Families are base operations, leakage-safe expected-behavior residuals, timestamp-level rack-peer aggregates, and optional causally aligned market context.

| Name | Role/family | Unit | Provenance | Availability |
|---|---|---|---|---|
| `site_max_thermal_residual_c`, `site_mean_thermal_residual_c` | Residual aggregate | Â°C | DERIVED | Rack residuals at/before T |
| `site_positive_thermal_residual_fraction` | Residual aggregate | fraction | DERIVED | Rack residuals at/before T |
| `site_max_temperature_peer_z` | Peer aggregate | dimensionless | DERIVED | Same-timestamp observable racks |
| `site_mean_temperature_peer_deviation_c` | Peer aggregate | Â°C | DERIVED | Same-timestamp observable racks |
| `site_max_voltage_spread_peer_z` | Peer aggregate | dimensionless | DERIVED | Same-timestamp observable racks |
| `site_fraction_racks_unavailable` | Peer/availability aggregate | fraction | DERIVED | At T |
| `failure_probability_6h`, `failure_probability_12h`, `failure_probability_24h` | Model output | probability | MODEL_PREDICTION | Target-free inference at T |

Targets are not predictors. `failure_within_6h`, `failure_within_12h`, and `failure_within_24h` mean a telemetry-defined failure occurs in `(T,T+h]`. Their provenance is `DERIVED FROM SIMULATED TELEMETRY`; incomplete horizons are `NaN`. Fault labels, future event IDs, and time-to-failure are evaluation-only and rejected from feature matrices.

## Availability engine outputs

All fields are `DERIVED ENGINEERING ANALYTIC`, not model predictions.

| Name | Entity | Definition | Unit/range |
|---|---|---|---|
| `technical_availability` | Site | available equal-sized racks / configured racks; unknown stays in denominator | fraction `[0,1]` |
| `rack_technical_availability`, `pcs_technical_availability` | Site | available rack/PCS count divided by configured count | fraction `[0,1]` |
| `known_component_fraction` | Site | known racks and PCS / installed components | fraction `[0,1]` |
| `available_discharge_power_mw`, `available_charge_power_mw` | Rack/PCS/site | observable directional AC power after component/thermal/converter limits | MW |
| `discharge_power_availability`, `charge_power_availability` | Site | directional MW / configured directional rating | fraction `[0,1]` |
| `available_discharge_energy_mwh` | Rack/PCS/site | SOH-adjusted energy above minimum SOC times discharge efficiency | deliverable AC MWh |
| `available_charge_energy_mwh` | Rack/PCS/site | SOH-adjusted headroom below maximum SOC divided by charge efficiency | acceptable AC MWh |
| `discharge_energy_availability`, `charge_energy_availability` | Site | directional energy / full SOC-window directional AC energy | fraction `[0,1]` |
| `available_power_mw` | Site | active-request direction; symmetric minimum while idle | MW |
| `available_energy_mwh` | Site | explicit alias for available discharge energy | deliverable AC MWh |
| `charge_headroom_mwh` | Site | alias for available charge energy | acceptable AC MWh |
| `requested_power_availability` | Site | `min(directional MW / abs(request), 1)`; null while idle | fraction `[0,1]` or null |
| `requested_energy_availability` | Site | energy sufficiency for configured one-hour request | fraction `[0,1]` or null |
| `sustainable_request_duration_hours` | Site | directional MWh / request MW | hours or null |
| `delivery_ratio` | Site | observed absolute actual/requested power; performance, not capability | ratio or null |
| `capability_delivery_gap` | Site | delivery ratio minus requested-power availability | diagnostic ratio |
| `limiting_factors`, `limiting_component_ids` | Component/site | observable constraints and contributors, not fault diagnosis | list[string] |

Legacy simulator `SiteTelemetry.available_energy_mwh` is stored DC energy above minimum SOC. Prompt 7 snapshot `available_energy_mwh` is deliverable AC discharge energy; provenance and schema distinguish them.

## Anomaly outputs and event metadata

| Name | Entity | Meaning | Type/provenance |
|---|---|---|---|
| `anomaly_score` | Rack/component | Detector-specific high-is-more-anomalous score; never a probability | float / DERIVED or MODEL_PREDICTION / ANOMALY MODEL OUTPUT |
| `anomaly_flag` | Rack/component interval | Validation-fixed threshold plus causal persistence decision | bool / DERIVED |
| `threshold`, `persistence_count` | Rack/component interval | Decision threshold and current consecutive run count | float, count / DERIVED |
| `supporting_signals`, `triggered_rules` | Rack/component interval | Transparent score/rule evidence | list[str] / DERIVED |
| `anomaly_event_id` | Component event | Deterministic ID after gap merging | string / DERIVED |
| `start_timestamp`, `end_timestamp`, `duration_minutes` | Component event | Event boundaries and duration | UTC, minutes / DERIVED |
| `peak_score`, `mean_score`, `trigger_count` | Component event | Within-event summary | float, count / DERIVED |
| `active_anomaly_count`, `number_racks_anomalous`, `number_pcs_anomalous` | Site | Transparent component roll-up | count / DERIVED |

Fault labels/IDs, severity, progression, future failure targets/episodes, delivery-risk probabilities, and SHAP values are not anomaly predictors. Ground truth is evaluation-only.

## Expected-behavior predictions and residual features

| Feature | Entity | Formula/source | Unit/type | Provenance | Causal/missing policy |
|---|---|---|---|---|---|
| `expected_actual_power_mw` | Site | Selected expected-power regressor | MW / float | MODEL_PREDICTION | Target not required; missing during insufficient walk-forward history |
| `power_residual_mw` | Site | actual power-expected power | MW / float | DERIVED | Available after actual arrives |
| `absolute_power_residual_mw` | Site | absolute power residual | MW / float | DERIVED | Follows residual availability |
| `normalized_power_residual` | Site | residual/rated power | fraction / float | DERIVED | Stable nameplate denominator |
| `delivery_shortfall_mw` | Site | `max(0, sign(requested)*(expected-actual))` | MW / float | DERIVED | Direction-correct for charge/discharge |
| `power_residual_rolling_mean`, `power_residual_rolling_std` | Site | trailing residual mean/sample standard deviation | MW / float | DERIVED | Causal configured residual window; std needs two rows |
| `negative_residual_fraction` | Site | trailing fraction residual `<0` | fraction / float | DERIVED | Diagnostic raw sign; direction differs for charging |
| `delivery_shortfall_rolling_mean` | Site | trailing mean directional shortfall | MW / float | DERIVED | Causal window |
| `residual_persistence_count` | Site | trailing count shortfall above configured threshold | count / float | DERIVED | Not an anomaly declaration |
| `expected_temperature_c` | Rack | Selected expected-temperature regressor | °C / float | MODEL_PREDICTION | Rack identity not required; missing during insufficient history |
| `thermal_residual_c` | Rack | actual mean temperature-expected temperature | °C / float | DERIVED | Positive means hotter than expected |
| `absolute_thermal_residual_c` | Rack | absolute thermal residual | °C / float | DERIVED | Follows residual availability |
| `thermal_residual_rolling_mean`, `thermal_residual_rolling_std` | Rack | trailing residual mean/sample standard deviation | °C / float | DERIVED | Causal configured residual window |
| `positive_thermal_residual_fraction` | Rack | trailing fraction residual `>0` | fraction / float | DERIVED | Not a threshold/alert |
| `thermal_residual_persistence_count` | Rack | trailing count residual above configured °C threshold | count / float | DERIVED | Not an anomaly declaration |
## Price-model features (`price_features_v1`)

| Feature | Definition | Availability |
|---|---|---|
| `price_lag_24h` | exact REAL price at T-24h | must exist by origin |
| `price_lag_168h` | exact REAL price at T-168h | must exist by origin |
| `rolling_price_mean_24h`, `rolling_price_mean_7d` | mean over `(T-24h-window,T-24h]` | causal for all targets through origin+24h |
| `price_volatility_24h`, `price_volatility_7d` | sample standard deviation over `(T-24h-window,T-24h]` | causal for all targets through origin+24h |
| `hour_of_day`, `day_of_week`, `hour_of_week`, `is_weekend` | Europe/Berlin target calendar | deterministic |
| `utc_offset_hours` | target's Europe/Berlin UTC offset | deterministic/DST disambiguation |
| `load_forecast_mw`, `wind_forecast_mw`, `solar_forecast_mw`, `renewable_forecast_mw` | optional ENTSO-E forecast for T | only with causal vintage |

The price target itself, future prices, future actual system measurements, battery fields,
availability, faults, anomalies, residual outcomes, and delivery-risk outputs are prohibited.
