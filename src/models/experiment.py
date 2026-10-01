"""Reproducible simulator-backed expected-behavior experiment entry point."""

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
from features.build import generate_feature_frames
from models.artifact import save_model_artifact
from models.expected_power.config import ExpectedPowerConfig
from models.expected_power.dataset import (
    build_expected_power_dataset,
    power_dataset_coverage,
)
from models.expected_power.evaluate import residual_fault_diagnostic
from models.expected_power.predict import add_power_residuals, predict_expected_power
from models.expected_power.train import train_expected_power
from models.expected_temperature.config import ExpectedTemperatureConfig
from models.expected_temperature.dataset import (
    build_expected_temperature_dataset,
    temperature_dataset_coverage,
)
from models.expected_temperature.evaluate import thermal_residual_fault_diagnostic
from models.expected_temperature.predict import (
    add_thermal_residuals,
    predict_expected_temperature,
)
from models.expected_temperature.train import (
    train_expected_temperature,
    unseen_rack_diagnostic,
)


def _faults(start: datetime, intervals: int, minutes: int) -> list[FaultEvent]:
    power_start = start + timedelta(minutes=minutes * int(intervals * 0.82))
    power_end = start + timedelta(minutes=minutes * int(intervals * 0.88))
    thermal_start = start + timedelta(minutes=minutes * int(intervals * 0.92))
    thermal_end = start + timedelta(minutes=minutes * int(intervals * 0.97))
    return [
        FaultEvent(
            fault_id="DEMO-POWER-TRACKING",
            fault_type=FaultType.POWER_TRACKING_ERROR,
            component_id="SITE",
            start_timestamp=power_start,
            end_timestamp=power_end,
            severity=0.45,
            progression_rate=0.0,
            ground_truth_label="POWER_TRACKING_ERROR",
        ),
        FaultEvent(
            fault_id="DEMO-THERMAL-DRIFT",
            fault_type=FaultType.THERMAL_DRIFT,
            component_id="PCS-01-RACK-01",
            start_timestamp=thermal_start,
            end_timestamp=thermal_end,
            severity=0.8,
            progression_rate=0.0,
            ground_truth_label="THERMAL_DRIFT",
        ),
    ]


def _offline_labels(
    simulator: BatterySimulator, timestamps: pd.Series, rack_ids: pd.Series | None = None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if rack_ids is None:
        for timestamp in timestamps:
            active = simulator.fault_engine.active_events(timestamp)
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "ground_truth_label": active[0].ground_truth_label if active else "NORMAL",
                    "fault_id": active[0].fault_id if active else None,
                    "fault_type": active[0].fault_type.value if active else "NORMAL",
                }
            )
        return pd.DataFrame(rows)
    for timestamp, rack_id in zip(timestamps, rack_ids, strict=True):
        rack_active = [
            event
            for event in simulator.fault_engine.active_events(timestamp)
            if event.component_id in {"SITE", rack_id}
            or (event.component_id.startswith("PCS-") and rack_id.startswith(event.component_id))
        ]
        rows.append(
            {
                "timestamp_utc": timestamp,
                "rack_id": rack_id,
                "ground_truth_label": (
                    rack_active[0].ground_truth_label if rack_active else "NORMAL"
                ),
                "fault_id": rack_active[0].fault_id if rack_active else None,
                "fault_type": rack_active[0].fault_type.value if rack_active else "NORMAL",
            }
        )
    return pd.DataFrame(rows)


def run_demo_experiment(
    output_root: str | Path,
    *,
    intervals: int = 360,
    fast: bool = False,
) -> dict[str, Any]:
    """Generate metrics from actual training; no values are hand-authored."""

    if intervals < 100:
        raise ValueError("Demo experiment requires at least 100 five-minute intervals")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    sim_config = SimulationConfig(
        features=FeatureConfig(require_full_windows=False), simulation_seed=42
    )
    faults = _faults(start, intervals, sim_config.battery.telemetry_interval_minutes)
    simulator = BatterySimulator(sim_config, faults)
    index = np.arange(intervals)
    requested = 18.0 * np.sin(2 * np.pi * index / 48)
    ambient = 18.0 + 7.0 * np.sin(2 * np.pi * index / 288)
    run = simulator.simulate(start, requested.tolist(), ambient.tolist())
    site_raw = pd.DataFrame([row.model_dump() for row in run.site_telemetry])
    rack_raw = pd.DataFrame([row.model_dump() for row in run.rack_telemetry])
    frames = generate_feature_frames(site_raw, rack_raw, config=sim_config)
    site = frames["site_features"]
    rack = frames["rack_features"]
    site_labels = _offline_labels(simulator, site["timestamp_utc"])
    rack_labels = _offline_labels(simulator, rack["timestamp_utc"], rack["rack_id"])
    site = site.merge(site_labels, on="timestamp_utc", validate="one_to_one")
    rack = rack.merge(rack_labels, on=["timestamp_utc", "rack_id"], validate="one_to_one")

    estimators = 20 if fast else 80
    power_config = ExpectedPowerConfig(
        minimum_training_rows=max(30, int(intervals * 0.25)),
        rf_n_estimators=estimators,
        lgbm_n_estimators=estimators,
    )
    temperature_config = ExpectedTemperatureConfig(
        minimum_training_rows=max(100, int(intervals * 4)),
        rf_n_estimators=estimators,
        lgbm_n_estimators=estimators,
    )
    power_dataset = build_expected_power_dataset(site, power_config)
    temperature_dataset = build_expected_temperature_dataset(rack, temperature_config)
    power_result = train_expected_power(power_dataset, power_config)
    temperature_result = train_expected_temperature(temperature_dataset, temperature_config)

    power_test_input = pd.concat(
        [power_result.split.test.keys, power_result.split.test.predictors], axis=1
    )
    power_predictions = predict_expected_power(power_result.artifact, power_test_input)
    power_actual = power_result.split.test.keys.copy()
    power_actual["actual_power_mw"] = power_result.split.test.target
    power_actual["requested_power_mw"] = power_result.split.test.predictors["requested_power_mw"]
    power_residuals = add_power_residuals(power_predictions, power_actual)
    power_residuals = pd.concat(
        [power_residuals, power_result.split.test.evaluation_metadata], axis=1
    )

    temperature_test_input = pd.concat(
        [temperature_result.split.test.keys, temperature_result.split.test.predictors],
        axis=1,
    )
    temperature_predictions = predict_expected_temperature(
        temperature_result.artifact, temperature_test_input
    )
    temperature_actual = temperature_result.split.test.keys.copy()
    temperature_actual["temperature_mean_c"] = temperature_result.split.test.target
    thermal_residuals = add_thermal_residuals(temperature_predictions, temperature_actual)
    thermal_residuals = pd.concat(
        [thermal_residuals, temperature_result.split.test.evaluation_metadata], axis=1
    )
    thermal_diagnostic_rows = thermal_residuals.loc[
        thermal_residuals["fault_type"].isin(["NORMAL", "THERMAL_DRIFT"])
    ]

    root = Path(output_root)
    power_dir = root / "expected_power"
    temperature_dir = root / "expected_temperature"
    save_model_artifact(power_result.artifact, power_dir)
    save_model_artifact(temperature_result.artifact, temperature_dir)
    power_residuals.to_parquet(power_dir / "test_predictions_residuals.parquet", index=False)
    thermal_residuals.to_parquet(
        temperature_dir / "test_predictions_residuals.parquet", index=False
    )
    (power_dir / "model_comparison.json").write_text(
        json.dumps(power_result.validation_comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temperature_dir / "model_comparison.json").write_text(
        json.dumps(temperature_result.validation_comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unseen = unseen_rack_diagnostic(
        temperature_dataset,
        "PCS-04-RACK-08",
        temperature_result.artifact.model_name,
        temperature_config,
    )
    summary: dict[str, Any] = {
        "simulation": {
            "intervals": intervals,
            "interval_minutes": sim_config.battery.telemetry_interval_minutes,
            "seed": sim_config.simulation_seed,
            "fault_events": len(faults),
        },
        "power": {
            "coverage": power_dataset_coverage(power_dataset),
            "selected_model": power_result.artifact.model_name,
            "validation_comparison": power_result.validation_comparison,
            "test_metrics": power_result.test_metrics,
            "fault_residual_diagnostic": residual_fault_diagnostic(power_residuals),
            "artifact_directory": str(power_dir),
        },
        "temperature": {
            "coverage": temperature_dataset_coverage(temperature_dataset),
            "selected_model": temperature_result.artifact.model_name,
            "validation_comparison": temperature_result.validation_comparison,
            "test_metrics": temperature_result.test_metrics,
            "fault_residual_diagnostic": thermal_residual_fault_diagnostic(thermal_diagnostic_rows),
            "unseen_rack_diagnostic": unseen,
            "artifact_directory": str(temperature_dir),
        },
    }
    (root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train expected power and rack-temperature demo models"
    )
    parser.add_argument("--output", default="artifacts/models")
    parser.add_argument("--intervals", type=int, default=360)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)
    summary = run_demo_experiment(args.output, intervals=args.intervals, fast=args.fast)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
