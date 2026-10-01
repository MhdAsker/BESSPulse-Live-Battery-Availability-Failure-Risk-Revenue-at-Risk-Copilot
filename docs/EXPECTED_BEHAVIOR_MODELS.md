# Expected-behavior model operation

Run the reproducible simulator-backed experiment:

```powershell
python -m models.experiment --output artifacts/models --intervals 180
```

This generates healthy training history, normal validation history, and a mixed normal/fault test period. It trains the naive, Ridge, Random Forest, and LightGBM candidates, selects on validation RMSE, evaluates test once, saves trusted-local artifacts, and writes prediction/residual Parquet files and machine-generated JSON comparisons.

Python inference uses `predict_expected_power` or `predict_expected_temperature` with the artifact feature contract. Targets are not required. Residual functions are separate and require subsequently observed actuals. Persisted predictions use `MODEL_PREDICTION`; residuals use `DERIVED`.

For leakage-safe historical residuals, call the walk-forward functions. They return missing predictions until minimum healthy history exists and record `training_max_timestamp`, which must be strictly earlier than the predicted timestamp.

Do not load untrusted joblib files. Do not interpret residuals as alerts, fault labels, or failure probabilities.

Optional diagnostic plots can be generated with `models.plots.save_regression_diagnostics`
after installing the `plots` extra. The helper produces predicted-vs-actual,
residual-vs-predicted, residual-over-time, residual-distribution, and optional
operating-mode/rack grouped figures in a caller-selected report directory; plot
generation is never part of normal tests.
