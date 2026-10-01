# Fault library

Faults use deterministic scheduled events with `fault_id`, `fault_type`, `component_id`, UTC start/end, severity in `[0,1]`, non-negative progression rate, and `ground_truth_label`. The active interval is `[start_timestamp, end_timestamp)`. Effective level is `min(1, severity + progression_rate × elapsed_hours)`.

| Fault | Component | Simulated effect at level `s` | Progression interpretation |
|---|---|---|---|
| `NORMAL` | Any | No change; useful as an explicit scenario marker | None |
| `PCS_DERATING` | PCS/site | Power factor `1-s` | Increasing converter derating |
| `RACK_OFFLINE` | Rack | Availability false and zero capability when active | Binary once level > 0 |
| `THERMAL_DRIFT` | Rack/site | Observed temperature offset up to +15 °C | Growing drift/hotspot indication |
| `COOLING_DEGRADATION` | Rack/PCS/site | Cooling effectiveness falls toward 10% | Worsening heat rejection |
| `SOC_SENSOR_BIAS` | Rack/site | Observed SOC offset up to +0.15; true SOC unchanged | Growing measurement bias |
| `VOLTAGE_IMBALANCE` | Rack/site | Voltage spread increases by up to 35 V | Growing imbalance proxy |
| `EFFICIENCY_DEGRADATION` | Rack/PCS/site | Efficiency multiplier falls to 0.75 | Increasing conversion loss |
| `ACCELERATED_CAPACITY_FADE` | Rack/PCS/site | Interval capacity fade multiplier rises to 101× | Synthetic accelerated ageing |
| `SELF_DISCHARGE` | Rack/PCS/site | Up to 0.5% of capacity per hour | Increasing parasitic loss |
| `POWER_TRACKING_ERROR` | Rack/PCS/site | Delivered/requested factor falls to 0.5 | Increasing control mismatch |

Site effects cascade to all racks; PCS effects cascade to child racks; rack effects remain local. Multiple active effects compose multiplicatively for factors and additively for offsets/rates.

These magnitudes are scenario assumptions rather than inferred empirical distributions. Events and point-in-time labels live in `ground_truth.py` and `fault_ground_truth`; telemetry contains symptoms only. `fault_type`, IDs, ground-truth labels, event boundaries, future state, and later future failure labels are forbidden predictive features.

## Prompt 6 expected versus measured anomaly behavior

Expected behavior remains a hypothesis: residual detectors may respond to thermal/power faults, peer voltage features may respond to imbalance, and availability may expose offline racks. The measured seed-42 test contains only one event per represented family.

| Fault family | Expected observable symptom | Measured test result |
|---|---|---|
| `THERMAL_DRIFT` | Higher temperature/residual | Detected at onset by engineering, Isolation Forest, residual, and vote; robust peer missed |
| `COOLING_DEGRADATION` | Sustained hotter-than-expected behavior | Detected at onset only by Isolation Forest |
| `VOLTAGE_IMBALANCE` | Higher spread/peer deviation | Detected at onset only by engineering; Isolation Forest and robust peer missed |
| `RACK_OFFLINE` | Availability/capability drop | Detected at onset only by Isolation Forest |
| `POWER_TRACKING_ERROR` | Direction-aware delivery shortfall | Detected at onset by Isolation Forest, residual, and vote |
| `PCS_DERATING` | PCS/rack delivery shortfall | Detected at onset only by Isolation Forest |

These are simulated single-event observations, not family-level performance estimates. No detector showed pre-fault warning.
