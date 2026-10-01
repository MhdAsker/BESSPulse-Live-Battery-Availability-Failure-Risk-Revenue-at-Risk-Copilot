# Delivery-risk model operation

Run the reproducible offline experiment:

```powershell
python -m models.delivery_risk.experiment --intervals 4032
```

No secret or live ENTSO-E request is required. The command simulates 14 days, generates causal operational and walk-forward expected-behavior features, derives censored 6/12/24-hour targets, applies horizon purge gaps, trains six comparisons, calibrates the selected classifier, evaluates test once, and writes artifacts and reports.

Artifacts live under `artifacts/models/delivery_risk/{6h,12h,24h}`. Reports live under `reports/delivery_risk/` and include comparisons, distributions, ablations, calibration bins, fault diagnostics, consistency measurements, and exact experiment summary. Joblib files must only be loaded from trusted local project outputs.

Python inference uses `predict_horizon` for one artifact or `predict_delivery_risk` for all three. Targets are not required. Missing horizon artifacts cause a controlled error rather than probability substitution. Outputs are `MODEL_PREDICTION`.

`global_feature_importance`, `global_tree_shap`, and `local_tree_shap` produce `DERIVED MODEL EXPLANATION` associations. They must not be described as physical or causal effects. Reliability figures are optional and never run in CI.
