"""Reproducible simulator-backed multi-horizon delivery-risk experiment."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import FeatureConfig
from besspulse.ground_truth import FaultEvent, FaultType
from features.battery import build_rack_features, build_site_features
from models.artifact import load_model_artifact, save_model_artifact
from models.delivery_risk.config import DeliveryRiskConfig
from models.delivery_risk.dataset import (
    RiskDataset,
    build_risk_dataset,
    dataset_distribution,
    purged_chronological_split,
)
from models.delivery_risk.evaluate import (
    classification_metrics,
    grouped_diagnostics,
    horizon_consistency,
    reliability_bins,
)
from models.delivery_risk.explain import (
    global_feature_importance,
    global_tree_shap,
    local_tree_shap,
)
from models.delivery_risk.predict import predict_delivery_risk, predict_horizon
from models.delivery_risk.targets import generate_delivery_failure_targets
from models.delivery_risk.train import candidate_classifiers, train_delivery_risk
from models.expected_power.config import ExpectedPowerConfig
from models.expected_power.dataset import build_expected_power_dataset
from models.expected_power.predict import add_power_residuals
from models.expected_power.train import walk_forward_expected_power
from models.expected_temperature.config import ExpectedTemperatureConfig
from models.expected_temperature.dataset import build_expected_temperature_dataset
from models.expected_temperature.predict import add_thermal_residuals
from models.expected_temperature.train import walk_forward_expected_temperature


def _demo_faults(start: datetime, hours: int) -> list[FaultEvent]:
    schedule = [
        (30, FaultType.POWER_TRACKING_ERROR, "SITE", 0.35),
        (65, FaultType.PCS_DERATING, "PCS-01", 0.55),
        (105, FaultType.POWER_TRACKING_ERROR, "SITE", 0.55),
        (135, FaultType.RACK_OFFLINE, "PCS-02-RACK-01", 0.8),
        (182, FaultType.POWER_TRACKING_ERROR, "SITE", 0.65),
        (190, FaultType.THERMAL_DRIFT, "PCS-01-RACK-01", 0.1),
        (226, FaultType.PCS_DERATING, "PCS-03", 0.75),
        (236, FaultType.POWER_TRACKING_ERROR, "SITE", 0.75),
        (273, FaultType.RACK_OFFLINE, "PCS-04-RACK-02", 1.0),
        (281, FaultType.POWER_TRACKING_ERROR, "SITE", 0.9),
        (300, FaultType.PCS_DERATING, "PCS-02", 0.9),
    ]
    events: list[FaultEvent] = []
    for index, (offset, fault_type, component, severity) in enumerate(schedule):
        if offset + 3 >= hours:
            continue
        event_start = start + timedelta(hours=offset)
        events.append(
            FaultEvent(
                fault_id=f"RISK-{index + 1:02d}",
                fault_type=fault_type,
                component_id=component,
                start_timestamp=event_start,
                end_timestamp=event_start + timedelta(hours=3),
                severity=severity,
                progression_rate=0.0,
                ground_truth_label=fault_type.value,
            )
        )
    return events


def _truth_at_timestamps(
    simulator: BatterySimulator, timestamps: pd.Series, entity_ids: pd.Series | None = None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, timestamp in enumerate(timestamps):
        entity_id = str(entity_ids.iloc[position]) if entity_ids is not None else "SITE"
        active = simulator.fault_engine.active_events(timestamp)
        events = (
            list(active)
            if entity_ids is None
            else [
                event
                for event in active
                if event.component_id in {"SITE", entity_id}
                or (
                    event.component_id.startswith("PCS-")
                    and entity_id.startswith(event.component_id)
                )
            ]
        )
        event = max(events, key=lambda item: item.severity) if events else None
        row: dict[str, Any] = {
            "timestamp_utc": timestamp,
            "ground_truth_label": event.ground_truth_label if event else "NORMAL",
            "fault_id": event.fault_id if event else None,
            "fault_type": event.fault_type.value if event else "NORMAL",
            "fault_severity": event.severity if event else 0.0,
        }
        if entity_ids is not None:
            row["rack_id"] = entity_id
        rows.append(row)
    return pd.DataFrame(rows)


def _rack_site_aggregates(rack: pd.DataFrame) -> pd.DataFrame:
    frame = rack.copy()
    grouped = frame.groupby("timestamp_utc", sort=False)
    median_temperature = grouped["temperature_mean_c"].transform("median")
    median_voltage = grouped["voltage_spread_v"].transform("median")
    temperature_mad = (
        (frame["temperature_mean_c"] - median_temperature)
        .abs()
        .groupby(frame["timestamp_utc"])
        .transform("median")
    )
    voltage_mad = (
        (frame["voltage_spread_v"] - median_voltage)
        .abs()
        .groupby(frame["timestamp_utc"])
        .transform("median")
    )
    frame["temperature_peer_deviation"] = frame["temperature_mean_c"] - median_temperature
    frame["temperature_peer_z"] = frame["temperature_peer_deviation"].div(
        (1.4826 * temperature_mad).clip(lower=1e-9)
    )
    frame["voltage_peer_z"] = (frame["voltage_spread_v"] - median_voltage).div(
        (1.4826 * voltage_mad).clip(lower=1e-9)
    )
    return (
        frame.groupby("timestamp_utc", sort=False)
        .agg(
            site_max_temperature_peer_z=("temperature_peer_z", lambda values: values.abs().max()),
            site_mean_temperature_peer_deviation_c=("temperature_peer_deviation", "mean"),
            site_max_voltage_spread_peer_z=("voltage_peer_z", lambda values: values.abs().max()),
            site_fraction_racks_unavailable=("availability", lambda values: 1 - values.mean()),
        )
        .reset_index()
    )


def _add_walk_forward_residuals(
    site: pd.DataFrame,
    rack: pd.DataFrame,
    simulator: BatterySimulator,
) -> pd.DataFrame:
    site_truth = _truth_at_timestamps(simulator, site["timestamp_utc"])
    site_labeled = site.merge(site_truth, on="timestamp_utc", validate="one_to_one")
    power_cfg = ExpectedPowerConfig(
        minimum_training_rows=120,
        walk_forward_block_timestamps=288,
        rf_n_estimators=20,
        lgbm_n_estimators=20,
    )
    power_dataset = build_expected_power_dataset(site_labeled, power_cfg)
    power_prediction = walk_forward_expected_power(
        power_dataset, model_name="ridge_linear", config=power_cfg
    )
    power_actual = site[["timestamp_utc", "asset_id", "actual_power_mw", "requested_power_mw"]]
    power_residual = add_power_residuals(power_prediction, power_actual)
    power_residual = power_residual.rename(
        columns={
            "power_residual_mw": "expected_power_residual_mw",
            "absolute_power_residual_mw": "expected_absolute_power_residual_mw",
            "power_residual_rolling_mean": "expected_power_residual_rolling_mean",
        }
    )
    power_columns = [
        "timestamp_utc",
        "asset_id",
        "expected_actual_power_mw",
        "expected_power_residual_mw",
        "expected_absolute_power_residual_mw",
        "delivery_shortfall_mw",
        "expected_power_residual_rolling_mean",
        "delivery_shortfall_rolling_mean",
        "residual_persistence_count",
    ]

    rack_truth = _truth_at_timestamps(simulator, rack["timestamp_utc"], rack["rack_id"])
    rack_labeled = rack.merge(rack_truth, on=["timestamp_utc", "rack_id"], validate="one_to_one")
    temperature_cfg = ExpectedTemperatureConfig(
        minimum_training_rows=1000,
        walk_forward_block_timestamps=576,
        rf_n_estimators=20,
        lgbm_n_estimators=20,
    )
    temperature_dataset = build_expected_temperature_dataset(rack_labeled, temperature_cfg)
    temperature_prediction = walk_forward_expected_temperature(
        temperature_dataset, model_name="naive", config=temperature_cfg
    )
    temperature_actual = rack[["timestamp_utc", "rack_id", "temperature_mean_c"]]
    thermal = add_thermal_residuals(temperature_prediction, temperature_actual)
    thermal_site = (
        thermal.groupby("timestamp_utc", sort=False)
        .agg(
            site_max_thermal_residual_c=("thermal_residual_c", "max"),
            site_mean_thermal_residual_c=("thermal_residual_c", "mean"),
            site_positive_thermal_residual_fraction=(
                "thermal_residual_c",
                lambda values: (values > 0).mean(),
            ),
        )
        .reset_index()
    )
    return (
        site.merge(power_residual[power_columns], on=["timestamp_utc", "asset_id"])
        .merge(_rack_site_aggregates(rack), on="timestamp_utc")
        .merge(thermal_site, on="timestamp_utc")
    )


def _event_count(dataset: RiskDataset) -> int:
    column = f"future_failure_event_ids_{dataset.horizon_hours}h"
    if column not in dataset.evaluation_metadata:
        return 0
    return len({event for events in dataset.evaluation_metadata[column] for event in events})


def _ablation(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
    config: DeliveryRiskConfig,
) -> list[dict[str, Any]]:
    designs = {
        "base": ("base",),
        "base_plus_residual": ("base", "residual"),
        "base_plus_residual_peer": ("base", "residual", "peer"),
    }
    output: list[dict[str, Any]] = []
    for name, families in designs.items():
        dataset = build_risk_dataset(features, targets, horizon, families=families)
        split = purged_chronological_split(dataset, config)
        estimator = candidate_classifiers(dataset, config)["logistic_regression"]
        estimator.fit(split.train.predictors, split.train.target.astype(int))
        probability = estimator.predict_proba(split.validation.predictors)[:, 1]
        metrics = classification_metrics(split.validation.target.astype(int), probability)
        output.append(
            {
                "horizon_hours": horizon,
                "feature_design": name,
                "status": "evaluated",
                "pr_auc_average_precision": metrics["pr_auc_average_precision"],
                "brier_score": metrics["brier_score"],
                "feature_count": len(dataset.predictors.columns),
            }
        )
    output.append(
        {
            "horizon_hours": horizon,
            "feature_design": "plus_real_entsoe_market",
            "status": "unavailable_no_real_entsoe_rows_in_offline_experiment",
            "pr_auc_average_precision": None,
            "brier_score": None,
            "feature_count": None,
        }
    )
    return output


def run_delivery_risk_experiment(
    output_root: str | Path = "artifacts/models/delivery_risk",
    report_root: str | Path = "reports/delivery_risk",
    *,
    intervals: int = 4032,
    fast: bool = False,
) -> dict[str, Any]:
    if intervals < 2304:
        raise ValueError("Experiment needs at least eight days for 24h purged partitions")
    start = datetime(2026, 2, 1, tzinfo=UTC)
    cfg = SimulationConfig(
        simulation_seed=42,
        features=FeatureConfig(require_full_windows=False),
    )
    hours = int(intervals * cfg.battery.telemetry_interval_minutes / 60)
    faults = _demo_faults(start, hours)
    simulator = BatterySimulator(cfg, faults)
    index = np.arange(intervals)
    request_magnitude = 1.8 + 0.15 * np.sin(2 * np.pi * index / 288)
    requested = np.where(index % 2 == 0, request_magnitude, -request_magnitude / 0.9216)
    requested[np.mod(index, 97) < 5] = 0.0
    ambient = 19 + 9 * np.sin(2 * np.pi * index / 288)
    run = simulator.simulate(start, requested.tolist(), ambient.tolist())
    site_raw = pd.DataFrame([row.model_dump() for row in run.site_telemetry])
    rack_raw = pd.DataFrame([row.model_dump() for row in run.rack_telemetry])
    site = build_site_features(site_raw, cfg)
    ambient_frame = site_raw[["timestamp_utc", "ambient_temperature_c"]]
    rack_raw = rack_raw.merge(ambient_frame, on="timestamp_utc", validate="many_to_one")
    rack = build_rack_features(rack_raw, cfg)
    features = _add_walk_forward_residuals(site, rack, simulator)

    targets = generate_delivery_failure_targets(site, DeliveryRiskConfig())
    truth = _truth_at_timestamps(simulator, targets["timestamp_utc"])
    targets = targets.merge(truth, on="timestamp_utc", validate="one_to_one")
    model_config = DeliveryRiskConfig(
        minimum_training_rows=80 if fast else 120,
        minimum_positive_rows=4 if fast else 12,
        minimum_failure_events=1 if fast else 2,
        rf_n_estimators=30 if fast else 100,
        lgbm_n_estimators=30 if fast else 100,
        xgb_n_estimators=30 if fast else 100,
    )
    output_directory = Path(output_root)
    report_directory = Path(report_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[int, Any] = {}
    summary: dict[str, Any] = {
        "simulation": {
            "intervals": intervals,
            "interval_minutes": cfg.battery.telemetry_interval_minutes,
            "duration_hours": hours,
            "seed": cfg.simulation_seed,
            "fault_events": len(faults),
            "market_context": "not loaded; no live ENTSO-E access",
        },
        "horizons": {},
    }
    ablations: list[dict[str, Any]] = []
    test_predictions: dict[int, pd.DataFrame] = {}
    fault_diagnostic_rows: list[dict[str, Any]] = []
    test_metrics_output: dict[str, Any] = {}
    calibration_output: dict[str, Any] = {}
    for horizon in model_config.horizons_hours:
        dataset = build_risk_dataset(features, targets, horizon)
        result = train_delivery_risk(dataset, model_config)
        directory = output_directory / f"{horizon}h"
        save_model_artifact(result.artifact, directory)
        prediction_input = pd.concat([result.split.test.keys, result.split.test.predictors], axis=1)
        prediction = predict_horizon(result.artifact, prediction_input)
        outcome = result.split.test.keys.copy()
        outcome[f"failure_within_{horizon}h"] = result.split.test.target.astype(int)
        prediction.merge(outcome, on=["timestamp_utc", "asset_id"]).to_parquet(
            directory / "test_predictions.parquet", index=False
        )
        pd.DataFrame(result.validation_comparison).T.to_csv(
            report_directory / f"model_comparison_{horizon}h.csv"
        )
        reliability = reliability_bins(
            result.split.test.target.astype(int),
            prediction[f"failure_probability_{horizon}h"].to_numpy(),
            model_config.calibration_bins,
        )
        (report_directory / f"reliability_{horizon}h.json").write_text(
            json.dumps(reliability, indent=2) + "\n", encoding="utf-8"
        )
        diagnostic_frame = prediction.merge(outcome, on=["timestamp_utc", "asset_id"])
        diagnostic_frame = pd.concat(
            [diagnostic_frame.reset_index(drop=True), result.split.test.evaluation_metadata], axis=1
        )
        diagnostic_frame = pd.concat(
            [diagnostic_frame, result.split.test.predictors.reset_index(drop=True)], axis=1
        )
        fault_diagnostic = grouped_diagnostics(
            diagnostic_frame,
            group_column="fault_type",
            target_column=f"failure_within_{horizon}h",
            probability_column=f"failure_probability_{horizon}h",
            minimum_rows=5,
        )
        fault_diagnostic_rows.extend({"horizon_hours": horizon, **row} for row in fault_diagnostic)
        diagnostic_frame["test_time_segment"] = pd.qcut(
            np.arange(len(diagnostic_frame)), 3, labels=["early", "middle", "late"]
        )
        temporal_stability = grouped_diagnostics(
            diagnostic_frame,
            group_column="test_time_segment",
            target_column=f"failure_within_{horizon}h",
            probability_column=f"failure_probability_{horizon}h",
            minimum_rows=model_config.minimum_subgroup_rows,
        )
        diagnostic_frame["soc_regime"] = pd.cut(
            diagnostic_frame["soc"],
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=["low_soc", "normal_soc", "high_soc"],
        )
        diagnostic_frame["power_regime"] = pd.cut(
            diagnostic_frame["requested_power_mw"].abs(),
            bins=[-np.inf, 1.0, 2.0, np.inf],
            labels=["low_request", "medium_request", "high_request"],
        )
        operating_diagnostics = {
            group: grouped_diagnostics(
                diagnostic_frame,
                group_column=group,
                target_column=f"failure_within_{horizon}h",
                probability_column=f"failure_probability_{horizon}h",
                minimum_rows=model_config.minimum_subgroup_rows,
            )
            for group in ["operating_mode", "soc_regime", "power_regime"]
        }
        high_severity = diagnostic_frame.loc[diagnostic_frame["fault_severity"] >= 0.8]
        unseen_severity = (
            classification_metrics(
                high_severity[f"failure_within_{horizon}h"].astype(int),
                high_severity[f"failure_probability_{horizon}h"].to_numpy(),
            )
            if len(high_severity) >= 5
            and high_severity[f"failure_within_{horizon}h"].nunique() == 2
            else {"status": "insufficient_mixed_outcomes"}
        )
        distribution = dataset_distribution(dataset)
        distribution["distinct_failure_events"] = _event_count(dataset)
        summary["horizons"][str(horizon)] = {
            "distribution": distribution,
            "selected_model": result.artifact.model_name,
            "calibration_method": result.artifact.metadata["calibration_method"],
            "validation_comparison": result.validation_comparison,
            "calibration_comparison": result.calibration_comparison,
            "test_metrics": result.test_metrics,
            "fault_family_diagnostic": fault_diagnostic,
            "temporal_stability": temporal_stability,
            "operating_regime_diagnostics": operating_diagnostics,
            "unseen_high_severity_diagnostic": unseen_severity,
            "held_out_rack_diagnostic": {
                "status": "not_applicable_site_classifier_has_no_rack_identity",
                "note": "rack signals are symmetric timestamp aggregates",
            },
            "artifact_directory": str(directory),
        }
        artifacts[horizon] = load_model_artifact(directory / "model.joblib")
        test_predictions[horizon] = prediction
        ablations.extend(_ablation(features, targets, horizon, model_config))
        test_metrics_output[str(horizon)] = result.test_metrics
        calibration_output[str(horizon)] = {
            "selected_method": result.artifact.metadata["calibration_method"],
            "comparison": result.calibration_comparison,
        }
        global_feature_importance(result.artifact).to_csv(
            directory / "native_feature_importance.csv", index=False
        )
        global_tree_shap(
            result.artifact,
            prediction_input,
            sample_size=model_config.shap_sample_size,
            random_seed=model_config.random_seed,
        ).to_csv(directory / "global_shap_importance.csv", index=False)
        (directory / "local_shap_example.json").write_text(
            json.dumps(local_tree_shap(result.artifact, prediction_input.iloc[[0]]), indent=2)
            + "\n",
            encoding="utf-8",
        )
    common_test = features[
        features["timestamp_utc"].isin(
            set.intersection(*(set(frame["timestamp_utc"]) for frame in test_predictions.values()))
        )
    ]
    combined = predict_delivery_risk(artifacts, common_test)
    consistency = horizon_consistency(combined)
    summary["horizon_consistency"] = consistency
    summary["feature_ablation"] = ablations
    pd.DataFrame(ablations).to_csv(report_directory / "feature_ablation.csv", index=False)
    pd.DataFrame(fault_diagnostic_rows).to_csv(
        report_directory / "fault_diagnostics.csv", index=False
    )
    (report_directory / "test_metrics.json").write_text(
        json.dumps(test_metrics_output, indent=2) + "\n", encoding="utf-8"
    )
    (report_directory / "calibration_metrics.json").write_text(
        json.dumps(calibration_output, indent=2) + "\n", encoding="utf-8"
    )
    (report_directory / "horizon_consistency.json").write_text(
        json.dumps(consistency, indent=2) + "\n", encoding="utf-8"
    )
    (report_directory / "target_distribution.json").write_text(
        json.dumps(
            {key: value["distribution"] for key, value in summary["horizons"].items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_directory / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run delivery-risk classification experiment")
    parser.add_argument("--output", default="artifacts/models/delivery_risk")
    parser.add_argument("--reports", default="reports/delivery_risk")
    parser.add_argument("--intervals", type=int, default=4032)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)
    summary = run_delivery_risk_experiment(
        args.output, args.reports, intervals=args.intervals, fast=args.fast
    )
    for horizon, result in summary["horizons"].items():
        metrics = result["test_metrics"]
        print(
            f"horizon={horizon}h model={result['selected_model']} "
            f"pr_auc={metrics['pr_auc_average_precision']:.4f} "
            f"brier={metrics['brier_score']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
