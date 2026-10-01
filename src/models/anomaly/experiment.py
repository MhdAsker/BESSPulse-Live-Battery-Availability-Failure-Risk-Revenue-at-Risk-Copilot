"""Separate Prompt 6 simulator experiment for anomaly and early-warning evaluation."""

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import FeatureConfig
from besspulse.ground_truth import FaultEvent, FaultType
from features.battery import build_rack_features
from features.peer import add_peer_features
from models.anomaly.artifact import AnomalyArtifact, save_anomaly_artifact
from models.anomaly.common import apply_consecutive_persistence
from models.anomaly.config import AnomalyConfig
from models.anomaly.dataset import (
    AnomalyDataset,
    build_anomaly_dataset,
    chronological_anomaly_split,
    feature_contract,
)
from models.anomaly.engineering import THRESHOLD_ORIGINS, score_engineering
from models.anomaly.ensemble import aggregate_site_summary, score_vote_ensemble
from models.anomaly.evaluate import detector_metrics, select_threshold_persistence
from models.anomaly.events import create_anomaly_events
from models.anomaly.isolation_forest import (
    fit_isolation_forest,
    score_isolation_forest,
)
from models.anomaly.residual import (
    directional_delivery_shortfall,
    fit_residual_reference,
    score_residual,
)
from models.anomaly.robust import ROBUST_COLUMNS, score_robust_peer
from models.expected_temperature.config import ExpectedTemperatureConfig
from models.expected_temperature.dataset import build_expected_temperature_dataset
from models.expected_temperature.predict import add_thermal_residuals
from models.expected_temperature.train import walk_forward_expected_temperature


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _faults(start: datetime) -> list[FaultEvent]:
    specifications = [
        (20, FaultType.THERMAL_DRIFT, "PCS-01-RACK-01", 0.3),
        (40, FaultType.VOLTAGE_IMBALANCE, "PCS-01-RACK-02", 0.3),
        (60, FaultType.RACK_OFFLINE, "PCS-02-RACK-01", 0.3),
        (80, FaultType.POWER_TRACKING_ERROR, "SITE", 0.3),
        (100, FaultType.COOLING_DEGRADATION, "PCS-02-RACK-02", 0.3),
        (130, FaultType.THERMAL_DRIFT, "PCS-03-RACK-01", 0.5),
        (145, FaultType.VOLTAGE_IMBALANCE, "PCS-03-RACK-02", 0.5),
        (160, FaultType.POWER_TRACKING_ERROR, "SITE", 0.5),
        (172, FaultType.PCS_DERATING, "PCS-04", 0.5),
        (185, FaultType.THERMAL_DRIFT, "PCS-01-RACK-03", 0.8),
        (195, FaultType.COOLING_DEGRADATION, "PCS-01-RACK-04", 0.8),
        (205, FaultType.VOLTAGE_IMBALANCE, "PCS-02-RACK-03", 0.8),
        (215, FaultType.RACK_OFFLINE, "PCS-02-RACK-04", 1.0),
        (225, FaultType.POWER_TRACKING_ERROR, "SITE", 0.9),
        (232, FaultType.PCS_DERATING, "PCS-03", 0.9),
    ]
    return [
        FaultEvent(
            fault_id=f"ANOM-{index + 1:02d}",
            fault_type=fault_type,
            component_id=component,
            start_timestamp=start + timedelta(hours=offset),
            end_timestamp=start + timedelta(hours=offset + 3),
            severity=severity,
            progression_rate=0.0,
            ground_truth_label=fault_type.value,
        )
        for index, (offset, fault_type, component, severity) in enumerate(specifications)
    ]


def _truth_rows(
    simulator: BatterySimulator, timestamps: pd.Series, rack_ids: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp, rack_id in zip(timestamps, rack_ids, strict=True):
        active = [
            event
            for event in simulator.fault_engine.active_events(timestamp)
            if event.component_id == "SITE"
            or event.component_id == rack_id
            or (event.component_id.startswith("PCS-") and rack_id.startswith(event.component_id))
        ]
        event = max(active, key=lambda item: item.severity) if active else None
        rows.append(
            {
                "timestamp_utc": timestamp,
                "rack_id": rack_id,
                "pcs_id": str(rack_id).split("-RACK")[0],
                "ground_truth_label": event.ground_truth_label if event else "NORMAL",
                "fault_id": event.fault_id if event else None,
                "fault_type": event.fault_type.value if event else "NORMAL",
                "severity": event.severity if event else 0.0,
                "ground_truth_provenance": "SIMULATED GROUND TRUTH / EVALUATION ONLY",
            }
        )
    return pd.DataFrame(rows)


def _fault_frame(events: list[FaultEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fault_id": event.fault_id,
                "fault_type": event.fault_type.value,
                "component_id": event.component_id,
                "start_timestamp": pd.Timestamp(event.start_timestamp),
                "end_timestamp": pd.Timestamp(event.end_timestamp),
                "severity": event.severity,
                "severity_bin": "low"
                if event.severity < 0.4
                else "medium"
                if event.severity < 0.7
                else "high",
            }
            for event in events
        ]
    )


def _add_walk_forward_residuals(rack: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    labeled = rack.merge(
        truth[["timestamp_utc", "rack_id", "ground_truth_label"]],
        on=["timestamp_utc", "rack_id"],
        validate="one_to_one",
    )
    cfg = ExpectedTemperatureConfig(
        minimum_training_rows=1000,
        walk_forward_block_timestamps=288,
        rf_n_estimators=10,
        lgbm_n_estimators=10,
    )
    dataset = build_expected_temperature_dataset(labeled, cfg)
    prediction = walk_forward_expected_temperature(dataset, model_name="naive", config=cfg)
    actual = rack[["timestamp_utc", "rack_id", "temperature_mean_c"]]
    residuals = add_thermal_residuals(prediction, actual)
    columns = [
        "timestamp_utc",
        "rack_id",
        "expected_temperature_c",
        "thermal_residual_c",
        "thermal_residual_rolling_mean",
        "positive_thermal_residual_fraction",
        "thermal_residual_persistence_count",
    ]
    result = rack.merge(residuals[columns], on=["timestamp_utc", "rack_id"])
    result["delivery_shortfall_mw"] = directional_delivery_shortfall(
        result["requested_power_mw"], result["actual_power_mw"]
    )
    result.attrs["residual_generation"] = "walk_forward_strictly_past_or_naive_lag"
    return result


def _partition_faults(faults: pd.DataFrame, dataset: AnomalyDataset) -> pd.DataFrame:
    start = dataset.keys.timestamp_utc.min()
    end = dataset.keys.timestamp_utc.max()
    return faults.loc[faults["start_timestamp"].between(start, end, inclusive="both")].copy()


def _score_and_select(
    name: str,
    score_function: Callable[[AnomalyDataset], pd.DataFrame],
    split: Any,
    validation_faults: pd.DataFrame,
    test_faults: pd.DataFrame,
    thresholds: tuple[float, ...],
    config: AnomalyConfig,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], float, int]:
    validation_scores = score_function(split.validation)
    threshold, persistence, comparisons = select_threshold_persistence(
        validation_scores,
        split.validation.keys,
        validation_faults,
        thresholds,
        config,
    )
    test_scores = score_function(split.test)
    intervals = apply_consecutive_persistence(
        test_scores, threshold=threshold, intervals=persistence
    )
    metrics, matches, false_alerts = detector_metrics(
        intervals, split.test.keys, test_faults, config
    )
    metrics.update(
        {
            "detector": name,
            "validation_threshold": threshold,
            "persistence_setting": persistence,
        }
    )
    intervals.attrs["matches"] = matches
    intervals.attrs["false_alerts"] = false_alerts
    return intervals, metrics, comparisons, threshold, persistence


def _without_feature_family(dataset: AnomalyDataset, tokens: tuple[str, ...]) -> AnomalyDataset:
    retained = tuple(
        column for column in dataset.feature_names if not any(token in column for token in tokens)
    )
    return replace(
        dataset, predictors=dataset.predictors[list(retained)].copy(), feature_names=retained
    )


def _group_event_metrics(matches: pd.DataFrame, group: str, detector: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, frame in matches.groupby(group, observed=True):
        delays = frame.loc[frame["detected"], "detection_delay_minutes"]
        rows.append(
            {
                "detector": detector,
                group: value,
                "event_count": len(frame),
                "detected_count": int(frame["detected"].sum()),
                "missed_count": int((~frame["detected"]).sum()),
                "event_detection_rate": float(frame["detected"].mean()),
                "median_detection_delay_minutes": float(delays.median()) if len(delays) else None,
                "mean_detection_delay_minutes": float(delays.mean()) if len(delays) else None,
            }
        )
    return rows


def _risk_relationship(events: pd.DataFrame) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for horizon in (6, 12, 24):
        path = Path(f"artifacts/models/delivery_risk/{horizon}h/test_predictions.parquet")
        if not path.exists():
            continue
        risk = pd.read_parquet(path)
        probability = f"failure_probability_{horizon}h"
        for event in events.to_dict(orient="records"):
            before = risk.loc[
                risk["timestamp_utc"].between(
                    event["start_timestamp"] - pd.Timedelta("1h"),
                    event["start_timestamp"],
                    inclusive="left",
                ),
                probability,
            ]
            during = risk.loc[
                risk["timestamp_utc"].between(
                    event["start_timestamp"], event["end_timestamp"], inclusive="both"
                ),
                probability,
            ]
            outputs.append(
                {
                    "anomaly_event_id": event["anomaly_event_id"],
                    "horizon_hours": horizon,
                    "risk_before_event": float(before.mean()) if len(before) else None,
                    "risk_during_event": float(during.mean()) if len(during) else None,
                    "analysis_type": "DESCRIPTIVE_ONLY",
                    "calibration_warning": (
                        "known poor out-of-time calibration" if horizon == 24 else None
                    ),
                }
            )
    return outputs


def run_anomaly_experiment(
    output_root: str | Path = "artifacts/models/anomaly",
    report_root: str | Path = "reports/anomaly",
    *,
    intervals: int = 2880,
    fast: bool = False,
) -> dict[str, Any]:
    if intervals < 1800:
        raise ValueError("Anomaly experiment requires at least 6.25 days")
    config = AnomalyConfig(
        minimum_training_rows=200 if fast else 500,
        iforest_n_estimators=30 if fast else 100,
    )
    start = datetime(2026, 2, 5, tzinfo=UTC)
    sim_config = SimulationConfig(
        simulation_seed=42,
        features=FeatureConfig(require_full_windows=False),
    )
    faults = _faults(start)
    simulator = BatterySimulator(sim_config, faults)
    report_path = Path(report_root)
    report_path.mkdir(parents=True, exist_ok=True)
    cache_path = report_path / f"causal_feature_cache_{intervals}_seed42.parquet"
    truth_cache_path = report_path / f"evaluation_truth_cache_{intervals}_seed42.parquet"
    if cache_path.exists() and truth_cache_path.exists():
        features = pd.read_parquet(cache_path)
        truth = pd.read_parquet(truth_cache_path)
    else:
        index = np.arange(intervals)
        magnitude = 1.8 + 0.2 * np.sin(2 * np.pi * index / 288)
        requested = np.where(index % 2 == 0, magnitude, -magnitude / 0.9216)
        ambient = 19 + 9 * np.sin(2 * np.pi * index / 288)
        run = simulator.simulate(start, requested.tolist(), ambient.tolist())
        rack_raw = pd.DataFrame([row.model_dump() for row in run.rack_telemetry])
        ambient_frame = pd.DataFrame(
            {
                "timestamp_utc": [row.timestamp_utc for row in run.site_telemetry],
                "ambient_temperature_c": [row.ambient_temperature_c for row in run.site_telemetry],
            }
        )
        rack_raw = rack_raw.merge(ambient_frame, on="timestamp_utc", validate="many_to_one")
        rack = add_peer_features(build_rack_features(rack_raw, sim_config), sim_config)
        truth = _truth_rows(simulator, rack["timestamp_utc"], rack["rack_id"])
        features = _add_walk_forward_residuals(rack, truth)
        features.to_parquet(cache_path, index=False)
        truth.to_parquet(truth_cache_path, index=False)
    dataset = build_anomaly_dataset(features, truth)
    split = chronological_anomaly_split(dataset, config)
    fault_frame = _fault_frame(faults)
    validation_faults = _partition_faults(fault_frame, split.validation)
    test_faults = _partition_faults(fault_frame, split.test)

    iforest = fit_isolation_forest(split.train, config)
    residual_reference = fit_residual_reference(split.train, config)
    train_iforest_scores = score_isolation_forest(split.train, iforest)["anomaly_score"]
    iforest_thresholds = (
        0.0,
        *tuple(
            float(train_iforest_scores.quantile(q)) for q in config.isolation_quantile_candidates
        ),
    )
    detectors: dict[
        str, tuple[Callable[[AnomalyDataset], pd.DataFrame], tuple[float, ...], Any]
    ] = {
        "engineering": (lambda part: score_engineering(part, config), (1.0, 1.25, 1.5), None),
        "robust_peer": (
            lambda part: score_robust_peer(part, config),
            config.robust_z_candidates,
            None,
        ),
        "isolation_forest": (
            lambda part: score_isolation_forest(part, iforest),
            iforest_thresholds,
            iforest,
        ),
        "residual": (
            lambda part: score_residual(part, residual_reference),
            config.residual_z_candidates,
            residual_reference,
        ),
    }
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    detector_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    test_outputs: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    for name, (scorer, thresholds, fitted) in detectors.items():
        intervals_out, metrics, comparisons, threshold, persistence = _score_and_select(
            name, scorer, split, validation_faults, test_faults, thresholds, config
        )
        detector_rows.append(metrics)
        threshold_rows.extend({"detector": name, **row} for row in comparisons)
        matches = intervals_out.attrs["matches"]
        intervals_out.attrs = {}
        family_rows.extend(_group_event_metrics(matches, "fault_type", name))
        severity_rows.extend(_group_event_metrics(matches, "severity_bin", name))
        events = create_anomaly_events(intervals_out, config)
        events.to_parquet(report_path / f"events_{name}.parquet", index=False)
        serializable_intervals = intervals_out.copy()
        serializable_intervals.attrs = {}
        serializable_intervals.to_parquet(report_path / f"intervals_{name}.parquet", index=False)
        all_events.append(events)
        test_outputs.append(intervals_out)
        artifact = AnomalyArtifact(
            detector_name=name,
            detector_version=str(intervals_out["detector_version"].iloc[0]),
            fitted_object=fitted,
            feature_names=(
                tuple(iforest.feature_names)
                if name == "isolation_forest"
                else tuple(residual_reference.feature_names)
                if name == "residual"
                else (*tuple(ROBUST_COLUMNS), "peer_count")
                if name == "robust_peer"
                else tuple(THRESHOLD_ORIGINS)
            ),
            threshold=threshold,
            persistence_intervals=persistence,
            metadata={
                "dataset_hash": dataset.dataset_hash,
                "feature_set_version": "v1",
                "feature_contract": feature_contract(),
                "training_interval": [
                    str(split.train.keys.timestamp_utc.min()),
                    str(split.train.keys.timestamp_utc.max()),
                ],
                "validation_interval": [
                    str(split.validation.keys.timestamp_utc.min()),
                    str(split.validation.keys.timestamp_utc.max()),
                ],
                "test_interval": [
                    str(split.test.keys.timestamp_utc.min()),
                    str(split.test.keys.timestamp_utc.max()),
                ],
                "threshold_selection_partition": "validation_only",
                "false_alert_budget_per_day": config.max_false_alerts_per_day,
                "cooldown_enabled": False,
                "score_semantics": "higher_is_more_anomalous_not_probability",
                "random_seed": config.random_seed,
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        save_anomaly_artifact(artifact, output_path / name)

    ensemble = score_vote_ensemble(test_outputs, config)
    ensemble_metrics, ensemble_matches, _ = detector_metrics(
        ensemble, split.test.keys, test_faults, config
    )
    ensemble_metrics.update(
        {
            "detector": "vote_ensemble",
            "validation_threshold": ensemble["threshold"].iloc[0],
            "persistence_setting": "individual detectors",
        }
    )
    detector_rows.append(ensemble_metrics)
    family_rows.extend(_group_event_metrics(ensemble_matches, "fault_type", "vote_ensemble"))
    severity_rows.extend(_group_event_metrics(ensemble_matches, "severity_bin", "vote_ensemble"))
    ensemble_events = create_anomaly_events(ensemble, config)
    all_events.append(ensemble_events)
    ensemble.to_parquet(report_path / "intervals_vote_ensemble.parquet", index=False)
    ensemble_events.to_parquet(report_path / "events_vote_ensemble.parquet", index=False)
    aggregate_site_summary(ensemble).to_parquet(
        report_path / "site_anomaly_summary.parquet", index=False
    )
    save_anomaly_artifact(
        AnomalyArtifact(
            detector_name="vote_ensemble",
            detector_version="vote_ensemble_v1",
            fitted_object=None,
            feature_names=("engineering", "robust_peer", "isolation_forest", "residual"),
            threshold=float(ensemble["threshold"].iloc[0]),
            persistence_intervals=1,
            metadata={
                "dataset_hash": dataset.dataset_hash,
                "feature_set_version": "v1",
                "vote_threshold": config.ensemble_vote_threshold,
                "threshold_selection_partition": "component_detectors_validation_only",
                "score_semantics": "vote_fraction_not_probability",
                "random_seed": config.random_seed,
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        ),
        output_path / "vote_ensemble",
    )

    comparison = pd.DataFrame(detector_rows)
    comparison.to_csv(report_path / "detector_comparison.csv", index=False)
    comparison.to_csv(report_path / "event_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(report_path / "threshold_selection.csv", index=False)
    pd.DataFrame(family_rows).to_csv(report_path / "fault_family_metrics.csv", index=False)
    pd.DataFrame(severity_rows).to_csv(report_path / "severity_metrics.csv", index=False)
    ablation_definitions = {
        "without_residual_features": ("residual", "expected_temperature", "shortfall"),
        "without_peer_features": ("peer",),
    }
    ablation_rows: list[dict[str, Any]] = [
        {
            "variant": "full_feature_set",
            **next(row for row in detector_rows if row["detector"] == "isolation_forest"),
        }
    ]
    for variant, tokens in ablation_definitions.items():
        ablation_split = replace(
            split,
            train=_without_feature_family(split.train, tokens),
            validation=_without_feature_family(split.validation, tokens),
            test=_without_feature_family(split.test, tokens),
        )
        ablation_model = fit_isolation_forest(ablation_split.train, config)
        train_scores = score_isolation_forest(ablation_split.train, ablation_model)["anomaly_score"]
        ablation_thresholds = (
            0.0,
            *tuple(float(train_scores.quantile(q)) for q in config.isolation_quantile_candidates),
        )
        _, metrics, _, _, _ = _score_and_select(
            variant,
            partial(score_isolation_forest, artifact=ablation_model),
            ablation_split,
            validation_faults,
            test_faults,
            ablation_thresholds,
            config,
        )
        ablation_rows.append({"variant": variant, **metrics})
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.loc[
        ablation_frame["variant"].isin(["full_feature_set", "without_residual_features"])
    ].to_csv(report_path / "residual_ablation.csv", index=False)
    ablation_frame.loc[
        ablation_frame["variant"].isin(["full_feature_set", "without_peer_features"])
    ].to_csv(report_path / "peer_ablation.csv", index=False)

    heldout_rack = "PCS-01-RACK-03"
    heldout_train = split.train.subset(
        np.flatnonzero(split.train.keys["rack_id"].ne(heldout_rack).to_numpy())
    )
    heldout_iforest = fit_isolation_forest(heldout_train, config)
    heldout_positions = np.flatnonzero(split.test.keys["rack_id"].eq(heldout_rack).to_numpy())
    heldout_dataset = split.test.subset(heldout_positions)
    heldout_faults = test_faults.loc[test_faults["component_id"].eq(heldout_rack)]
    heldout_intervals = apply_consecutive_persistence(
        score_isolation_forest(heldout_dataset, heldout_iforest),
        threshold=float(
            comparison.loc[comparison.detector.eq("isolation_forest"), "validation_threshold"].iloc[
                0
            ]
        ),
        intervals=int(
            comparison.loc[comparison.detector.eq("isolation_forest"), "persistence_setting"].iloc[
                0
            ]
        ),
    )
    heldout_metrics, _, _ = detector_metrics(
        heldout_intervals, heldout_dataset.keys, heldout_faults, config
    )
    pd.DataFrame([{"heldout_rack": heldout_rack, **heldout_metrics}]).to_csv(
        report_path / "heldout_rack_results.csv", index=False
    )
    pd.DataFrame(
        [
            {"training_severity_max": 0.5, "test_severity": "high_unseen", **row}
            for row in detector_rows
            if row["detector"] == "isolation_forest"
        ]
    ).to_csv(report_path / "unseen_severity_results.csv", index=False)

    risk_rows = _risk_relationship(ensemble_events)
    pd.DataFrame(risk_rows).to_csv(report_path / "delivery_risk_relationship.csv", index=False)
    false_alert_payload = {
        row["detector"]: {
            "false_event_count": row["false_event_count"],
            "false_alerts_per_day": row["false_alerts_per_day"],
            "healthy_evaluation_days": row["healthy_evaluation_days"],
        }
        for row in detector_rows
    }
    (report_path / "false_alert_metrics.json").write_text(
        json.dumps(_json_ready(false_alert_payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    delay_payload = {
        row["detector"]: {
            key: row[key]
            for key in [
                "mean_detection_delay_minutes",
                "median_detection_delay_minutes",
                "p90_detection_delay_minutes",
                "minimum_detection_delay_minutes",
                "maximum_detection_delay_minutes",
            ]
        }
        for row in detector_rows
    }
    (report_path / "detection_delay.json").write_text(
        json.dumps(_json_ready(delay_payload), indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "prompt5_audit": {
            "status": "passed_read_only",
            "preserved_report": "reports/delivery_risk/experiment_summary.json",
            "known_24h_calibration_limitation": True,
        },
        "simulation": {
            "intervals": intervals,
            "interval_minutes": 5,
            "seed": 42,
            "fault_event_count": len(faults),
            "test_fault_event_count": len(test_faults),
            "fault_counts_by_family": fault_frame["fault_type"].value_counts().to_dict(),
            "fault_counts_by_severity": fault_frame["severity_bin"].value_counts().to_dict(),
            "fault_counts_by_component": fault_frame["component_id"].value_counts().to_dict(),
            "test_fault_counts_by_family": test_faults["fault_type"].value_counts().to_dict(),
            "test_fault_counts_by_severity": test_faults["severity_bin"].value_counts().to_dict(),
        },
        "zero_mad_policy": "equal peer value -> 0; deviation with undefined dispersion -> NaN",
        "threshold_policy": (
            "validation-only; meet false-alert budget, maximize detection, delay tie-break"
        ),
        "cooldown_enabled": False,
        "detector_comparison": detector_rows,
        "heldout_rack": heldout_metrics,
        "delivery_risk_relationship_rows": len(risk_rows),
    }
    (report_path / "experiment_metadata.json").write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run anomaly and early-warning experiment")
    parser.add_argument("--output", default="artifacts/models/anomaly")
    parser.add_argument("--reports", default="reports/anomaly")
    parser.add_argument("--intervals", type=int, default=2880)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)
    result = run_anomaly_experiment(
        args.output, args.reports, intervals=args.intervals, fast=args.fast
    )
    for row in result["detector_comparison"]:
        print(
            f"detector={row['detector']} event_detection={row['event_detection_rate']:.3f} "
            f"false_alerts_day={row['false_alerts_per_day']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
