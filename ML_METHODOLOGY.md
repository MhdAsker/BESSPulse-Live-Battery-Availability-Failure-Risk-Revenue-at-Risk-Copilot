# BESSPulse ML methodology

## Scientific objective

These regressors estimate healthy expected site power and rack temperature under observable operating conditions. They are not fault, anomaly, or delivery-risk classifiers. Predictions are `MODEL_PREDICTION`; actual-minus-prediction residuals and evaluation metrics are `DERIVED`.

## Training cohort and leakage boundary

Offline training requires an explicit `ground_truth_label`. Rows labeled `NORMAL`, `normal`, or `healthy` are eligible when the asset/rack is observably available, predictors and target are usable, and—in the power model—the request exceeds the configured active threshold. Ground truth is retained only in evaluation metadata and never enters `X`. Production feature generation and inference require no fault label.

Expected-power predictors are requested power and magnitude, SOC, ambient temperature, available rack/PCS counts and fractions, available power, recent alarm count/frequency, RTE, and non-ordinal operating mode. Expected-temperature predictors are rack actual/requested power and magnitude, ambient temperature, SOC, exact 5/15-minute temperature lags, shifted 30-minute temperature mean, 30-minute absolute power, prior-temperature-minus-ambient cooling proxy, throughput change, and operating state. Rack and PCS IDs are metadata only.

## Chronology and preprocessing

Unique timestamps are split 60% train, 20% validation, and 20% test. All racks at one timestamp remain together. If a known `fault_id` crosses a proposed boundary, the boundary moves to the event start where feasible. No row shuffle or random K-fold is used.

Median imputation, scaling, and one-hot vocabulary are fit only on healthy rows in the train partition. Validation and test use those fitted transforms. Unknown operating modes are safely ignored by the encoder. Invalid numeric strings and infinities fail inference; legitimate missing numeric values use the train-fitted median.

## Candidates and selection

Each task compares:

1. Naive physical baseline: requested power for power; exact 5-minute lag for temperature.
2. Ridge-stabilized linear regression with scaled numeric and one-hot categorical inputs.
3. Random Forest with bounded depth and deterministic seed.
4. LightGBM with a small fixed configuration, one thread, and deterministic settings.

Ordinary least squares was replaced by Ridge (`alpha=1`) after an actual reproducible run showed severe out-of-range thermal extrapolation caused by correlated lag/power predictors. Ridge remains an interpretable linear baseline while controlling conditioning.

Models are fitted on healthy train rows. Selection uses healthy validation RMSE only: choose the simplest candidate within 2% of the best validation RMSE. Test data are evaluated once after selection and do not influence the choice. MAE, RMSE, R², residual bias, and P95 absolute residual are recorded; power additionally reports active-interval MAE/RMSE. MAPE is intentionally omitted around zero power.

## Residual semantics

Power residual is `actual - expected`. With positive discharge and negative charge, under-delivery has opposite raw-residual signs by direction. Direction-neutral shortfall is:

```text
max(0, sign(requested) * (expected - actual))
```

Thermal residual is `actual temperature - expected temperature`; positive values mean hotter than expected. Causal trailing residual mean/std, directional/positive fractions, shortfall mean, and persistence counts use only values at or before the timestamp.

## Walk-forward historical residuals

Historical predictions begin only after the configured minimum healthy rows. For each chronological block, an estimator is fitted exclusively on healthy rows with timestamps strictly before the block. Early rows remain unavailable. This prevents generating later-model features from an expected-behavior model fitted on the same future period.

## Latest reproducible experiment

`python -m models.experiment --output artifacts/models --intervals 180` generated 15 hours at five-minute resolution using simulator seed 42, 180 site rows, 5,760 rack rows, 32 racks, and two faults reserved inside the test period with normal gaps around them. Machine-generated comparisons and exact metrics are stored under `artifacts/models/`; summarized results are in [MODEL_CARD.md](MODEL_CARD.md).

## Limitations and future use

The expected-behavior experiment is short, synthetic, single-asset, and not production validation. Fault labels define only its offline healthy cohort and diagnostics. Residual separation and anomaly scoring are handled by the later dedicated anomaly methodology. Model-native coefficients/importances are associative diagnostics, not causal effects.

## Delivery-risk classification

Delivery failure is derived from active requested-versus-actual power telemetry, never from fault presence. The targets search `(T, T+h]` for `h=6,12,24h`; incomplete end horizons are censored. Separate horizon datasets use 55/15/15/15 train/validation/calibration/test blocks. A horizon-sized purge removes every preceding row whose label window could reach the next partition.

The curated manifest groups causal predictors into base operational telemetry, leakage-safe expected-behavior residuals, timestamp-level rack peer aggregates, and optional market context. Target/truth/future fields and rack IDs are rejected. Median imputation, scaling, and one-hot encoding fit on train only. Class imbalance uses balanced weights where supported; no SMOTE, random split, or shuffled CV is used.

Candidates are Logistic Regression, Random Forest, LightGBM, and XGBoost, compared with constant prevalence and a fixed engineering score. Selection uses validation average precision, a fixed 0.01 effective-tie tolerance, Brier score, and fixed simplicity order. Identity, Platt, and isotonic calibration fit only on the later calibration block; the lowest calibration-block Brier score is fixed before test evaluation.

Metrics include precision, recall, F1, ROC-AUC, average precision, Brier score, expected calibration error, confusion matrix, and prevalence. Reliability bins, time segments, operating regimes, fault families, unseen severity, feature ablations, and horizon violations are diagnostics. SHAP and native importance indicate predictive association, not causality.

The latest run used 4,032 five-minute intervals (14 days), seed 42, 11 diverse faults, and eight telemetry-derived failure episodes. No real ENTSO-E rows were fetched, so market ablation is explicitly unavailable. The machine-generated record is `reports/delivery_risk/experiment_summary.json`.

## Anomaly detection and early warning

Anomaly detection is distinct from expected-behavior regression and future delivery-failure classification. Rack rows use curated observable features available at or before T. Fault type/ID, severity, progression, delivery targets, future episode data, delivery-risk probabilities, and SHAP values are prohibited from X. Simulated ground truth selects the healthy offline cohort and is then used only for validation tuning and held-out evaluation.

The 10-day seed-42 experiment uses chronological 50/25/25 train/validation/test partitions. Learned preprocessing and Isolation Forest fit on healthy train rows only. Residual centers and MAD scales fit on healthy train only; historical thermal expectations use strictly past walk-forward/lag information. Threshold and causal persistence are selected on validation under a four-false-alert/day target. If no candidate meets that budget, the predeclared detection/delay ordering is used and the violation is reported.

Engineering rules retain trigger and threshold origin. Robust peer score is the maximum valid absolute canonical leave-one-out z-score. For MAD at or below epsilon, equality yields zero and a nonzero deviation yields `NaN`; infinity or invented large scores are forbidden. Isolation Forest negates scikit-learn `decision_function`, so higher project score is more anomalous. Residual score uses train-only robust references, positive hotter-than-expected thermal deviation, and direction-aware delivery shortfall. The ensemble is an explicit two-of-four vote.

Flags merge by component/detector for gaps at most 10 minutes. Evaluation assigns early detections in `[fault start - 2h, fault start)`, onset at zero delay, and late detections through fault end plus 30 minutes. Unmatched events are false alerts. The denominator is elapsed test duration minus the union of fault-active time, not row count. Missed faults retain `NaN` delay.

Isolation Forest detected 5/6 test events with 17.1769 false alerts/day. Engineering, residual, and vote each detected 2/6 with zero false alerts/day; robust peer detected none. Every matched detection began at formal onset, so mean and median delay were zero among detected faults; no pre-fault warning was demonstrated. Prompt 5 probabilities are used only after scoring for descriptive comparison. The 24-hour score has severe known out-of-time calibration error and is not a reliable absolute risk probability.

## Availability interaction with models

Availability is not an ML task. It is calculated first from observable component state, temperature, SOC, simulated SOH proxy, configured limits, and converter/rack conservation. It imports no expected-behavior, delivery-risk, or anomaly model.

Frozen Prompt 5 probabilities and Prompt 6 events are timestamp-joined only after snapshots are complete. In the measured run, capability supported all modest requests but observed delivery failed on 3.75% of active intervals, demonstrating information beyond a simple physical limit calculation. All risk comparisons are descriptive; the 24-hour calibration limitation remains explicit.
## DE-LU price forecasting

Prompt 8 uses chronological 60/20/20 partitions. Lag-24, lag-168, train-only hour-of-week median,
a simple blend, and deterministic LightGBM are compared by validation MAE. A simpler model wins if
it is within 2% of the best; test is touched only after that decision. Expanding-window daily
backtests retrain strictly before each origin. Training-only P95 defines the high-price regime.
Negative prices and spikes are preserved. Separate P10/P50/P90 LightGBM models expose raw crossing,
pinball loss, and empirical coverage; row sorting is documented monotonic post-processing, not a
substitute for calibration.
