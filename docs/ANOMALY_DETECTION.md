# Anomaly detection and early fault warning

## Scope and contracts

Prompt 6 compares four rack-level detectors and a transparent two-of-four vote. Scores are not probabilities. All BESS telemetry is `SIMULATED`; fault schedules are `SIMULATED GROUND TRUTH / EVALUATION ONLY`. ENTSO-E data elsewhere are `REAL`, but this experiment makes no live market query.

Detector X contains only curated observable/causal operations, Prompt 3 canonical leave-one-out peer features, and Prompt 4-style leakage-safe residuals. It rejects fault labels/IDs, severity/progression, future state, delivery targets/episode IDs, time-to-failure, delivery-risk probabilities, and SHAP. Rack/PCS identity stays in keys, never in Isolation Forest X.

## Detectors

- Engineering: configured physical/engineering thresholds with triggered rule, origin, severity, and supporting signal retained.
- Robust peer: maximum absolute valid canonical robust z-score. If MAD is at most epsilon, equality maps to zero and a nonzero deviation maps to `NaN`; no infinity or fabricated large score is allowed.
- Isolation Forest: healthy simulated truth selects train rows offline. Median imputation and robust scaling fit on train only. Project score is `-decision_function`, so higher means more anomalous.
- Residual: healthy-train medians/MADs normalize positive thermal residual and direction-aware delivery shortfall. Historical temperature expectations use strictly past information.
- Ensemble: explicit two-of-four vote; individual outputs remain available.

## Time and event policy

The 2,880-interval seed-42 run uses 50% train, 25% validation, and 25% test in time order. Fault crossings move boundaries to event start. Validation alone chooses thresholds and consecutive-interval persistence. The candidate false-alert budget is four events per healthy day; violations are reported rather than hidden.

Flags for one component/detector merge when their gap is at most 10 minutes. Evaluation-only matching counts an event as early within two hours before fault start, at onset for zero delay, or late through fault end plus 30 minutes. Delay is first matched event start minus fault start. Missed faults retain `NaN`. False alerts are unmatched events divided by elapsed test duration after subtracting the union of fault-active time. Cooldown is disabled.

## Measured experiment

The simulation contains 15 faults: five low, four medium, and six high severity. Test has six high-severity faults—one each for thermal drift, cooling degradation, voltage imbalance, rack offline, power tracking error, and PCS derating. Each family therefore has only one test event.

| Detector | Detected | Missed | False events | False alerts/day | Mean delay min | Median delay min |
|---|---:|---:|---:|---:|---:|---:|
| Engineering | 2/6 | 4 | 0 | 0.0000 | 0 | 0 |
| Robust peer | 0/6 | 6 | 0 | 0.0000 | N/A | N/A |
| Isolation Forest | 5/6 | 1 | 30 | 17.1769 | 0 | 0 |
| Residual | 2/6 | 4 | 0 | 0.0000 | 0 | 0 |
| Vote ensemble | 2/6 | 4 | 0 | 0.0000 | 0 | 0 |

Isolation Forest detected every represented family except voltage imbalance. Engineering detected thermal drift and voltage imbalance. Residual and vote detected thermal drift and power-tracking error. All detections began at onset; there was no measured early warning. Isolation Forest's detection rate is offset by an unacceptable false-alert rate.

Held-out rack `PCS-01-RACK-03` was excluded from Isolation Forest training. Its one fault was detected at onset, with two false events and 0.8433 false alerts/day over 2.3715 healthy days. The unseen-high-severity diagnostic detected 5/6 but still produced 17.1769 false alerts/day.

With the same split and tuning, removing residual features preserved 5/6 detection, worsened false alerts from 17.1769 to 19.4672/day, and changed median delay from 0 to 5 minutes. Removing peer features preserved 5/6, reduced false alerts to 13.7416/day, and retained zero median delay. These are diagnostics, not causal claims.

## Delivery-risk relationship and limitations

`reports/anomaly/delivery_risk_relationship.csv` joins anomaly events to frozen Prompt 5 6/12/24-hour probabilities only after detector evaluation. It is descriptive and cannot establish causality. The Prompt 5 24-hour model is severely miscalibrated out of time, so its values are not reliable absolute risks.

This is one simulated asset with six test faults and one event per family. Abrupt fault starts limit the opportunity for early detection. Identical simulated peers make dispersion undefined for many genuine deviations, explaining weak robust-peer behavior. There is no real-site or commercial threshold validation. Do not use these artifacts for safety, control, availability guarantees, diagnosis, alert priority, or commercial decisions.
