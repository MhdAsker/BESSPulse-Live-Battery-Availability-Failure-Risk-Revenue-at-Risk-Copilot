"""Reproducible Prompt 7 availability experiment and contextual diagnostics."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from availability.config import AvailabilityConfig
from availability.engine import calculate_availability_bundle
from availability.metrics import (
    availability_summary,
    component_downtime,
    create_availability_events,
    interval_weights_hours,
    time_weighted_mean,
)
from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import FeatureConfig
from models.anomaly.experiment import _faults


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _time_fraction(mask: pd.Series, timestamps: pd.Series) -> float:
    weights = interval_weights_hours(timestamps)
    total = float(weights.sum())
    return float(weights.loc[mask].sum() / total) if total > 0 else float("nan")


def _delivery_risk_context(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (6, 12, 24):
        path = Path(f"artifacts/models/delivery_risk/{horizon}h/test_predictions.parquet")
        if not path.exists():
            continue
        risk = pd.read_parquet(path)
        probability = f"failure_probability_{horizon}h"
        columns = ["timestamp_utc", probability]
        context = snapshots.merge(risk[columns], on="timestamp_utc", how="inner")
        for dimension in (
            "technical_availability",
            "power_availability",
            "discharge_energy_availability",
        ):
            bins = pd.cut(
                context[dimension],
                bins=[-1e-9, 0.8, 0.95, 0.999999, 1.000001],
                labels=["<=0.80", "0.80-0.95", "0.95-<1", "~1"],
                include_lowest=True,
            )
            for availability_bin, frame in context.groupby(bins, observed=True):
                rows.append(
                    {
                        "horizon_hours": horizon,
                        "availability_dimension": dimension,
                        "availability_bin": str(availability_bin),
                        "row_count": len(frame),
                        "mean_failure_probability": float(frame[probability].mean()),
                        "analysis_type": "DESCRIPTIVE_ONLY",
                        "calibration_warning": (
                            "known poor out-of-time calibration" if horizon == 24 else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _anomaly_context(snapshots: pd.DataFrame) -> dict[str, Any]:
    path = Path("reports/anomaly/events_vote_ensemble.parquet")
    if not path.exists():
        return {"status": "unavailable", "reason": "Prompt 6 event report missing"}
    events = pd.read_parquet(path)
    reduced = snapshots.loc[
        snapshots["technical_availability"].lt(1 - 1e-6)
        | snapshots["power_availability"].lt(1 - 1e-6)
    ]
    categories = {
        "anomaly_before_availability_loss": 0,
        "anomaly_during_availability_loss": 0,
        "anomaly_without_availability_loss": 0,
    }
    for event in events.to_dict(orient="records"):
        start = pd.Timestamp(event["start_timestamp"])
        end = pd.Timestamp(event["end_timestamp"])
        during = reduced["timestamp_utc"].between(start, end, inclusive="both").any()
        after = (
            reduced["timestamp_utc"].between(end, end + pd.Timedelta("2h"), inclusive="right").any()
        )
        if during:
            categories["anomaly_during_availability_loss"] += 1
        elif after:
            categories["anomaly_before_availability_loss"] += 1
        else:
            categories["anomaly_without_availability_loss"] += 1
    reduced_flag = snapshots.index.isin(reduced.index)
    reduction_starts = snapshots.loc[
        reduced_flag & ~pd.Series(reduced_flag, index=snapshots.index).shift(fill_value=False),
        "timestamp_utc",
    ]
    without_prior_anomaly = 0
    for reduction_start in reduction_starts:
        prior = (
            events["start_timestamp"]
            .between(reduction_start - pd.Timedelta("2h"), reduction_start, inclusive="both")
            .any()
        )
        if not prior:
            without_prior_anomaly += 1
    reduction_count = len(reduced)
    return {
        "status": "evaluated",
        "analysis_type": "DESCRIPTIVE_ONLY",
        "anomaly_event_count": len(events),
        "availability_reduction_intervals": reduction_count,
        "availability_loss_event_count": len(reduction_starts),
        "availability_loss_without_prior_anomaly": without_prior_anomaly,
        **categories,
    }


def run_availability_experiment(
    report_root: str | Path = "reports/availability",
    *,
    intervals: int = 2880,
) -> dict[str, Any]:
    start = datetime(2026, 2, 5, tzinfo=UTC)
    simulation_config = SimulationConfig(
        simulation_seed=42,
        features=FeatureConfig(require_full_windows=False),
    )
    config = AvailabilityConfig(simulation=simulation_config)
    faults = _faults(start)
    index = np.arange(intervals)
    magnitude = 1.8 + 0.2 * np.sin(2 * np.pi * index / 288)
    requested = np.where(index % 2 == 0, magnitude, -magnitude / 0.9216)
    ambient = 19 + 9 * np.sin(2 * np.pi * index / 288)
    run = BatterySimulator(simulation_config, faults).simulate(
        start, requested.tolist(), ambient.tolist()
    )
    site = pd.DataFrame(row.model_dump() for row in run.site_telemetry)
    pcs = pd.DataFrame(row.model_dump() for row in run.pcs_telemetry)
    racks = pd.DataFrame(row.model_dump() for row in run.rack_telemetry)
    bundle = calculate_availability_bundle(site, pcs, racks, config)
    report_path = Path(report_root)
    report_path.mkdir(parents=True, exist_ok=True)
    bundle.snapshots.to_parquet(report_path / "availability_snapshots.parquet", index=False)
    bundle.rack_capabilities.to_parquet(report_path / "rack_capabilities.parquet", index=False)
    bundle.pcs_capabilities.to_parquet(report_path / "pcs_capabilities.parquet", index=False)
    rack_downtime = component_downtime(bundle.rack_capabilities)
    pcs_downtime = component_downtime(bundle.pcs_capabilities)
    rack_downtime.to_csv(report_path / "rack_downtime.csv", index=False)
    pcs_downtime.to_csv(report_path / "pcs_downtime.csv", index=False)
    rack_events = create_availability_events(bundle.rack_capabilities)
    pcs_events = create_availability_events(bundle.pcs_capabilities)
    rack_events.to_parquet(report_path / "rack_availability_events.parquet", index=False)
    pcs_events.to_parquet(report_path / "pcs_availability_events.parquet", index=False)
    risk_context = _delivery_risk_context(bundle.snapshots)
    risk_context.to_csv(report_path / "delivery_risk_context.csv", index=False)
    anomaly_context = _anomaly_context(bundle.snapshots)
    _write_json(report_path / "anomaly_context.json", anomaly_context)
    truth = pd.DataFrame(row.model_dump() for row in run.fault_ground_truth)
    if not truth.empty:
        truth["fault_type"] = truth["fault_type"].astype(str)
        diagnostic = bundle.snapshots.merge(
            truth[["timestamp_utc", "fault_type"]].drop_duplicates(),
            on="timestamp_utc",
            how="inner",
        )
        fault_diagnostics = (
            diagnostic.groupby("fault_type", as_index=False)
            .agg(
                interval_count=("timestamp_utc", "size"),
                mean_technical_availability=("technical_availability", "mean"),
                mean_power_availability=("power_availability", "mean"),
                mean_discharge_energy_availability=(
                    "discharge_energy_availability",
                    "mean",
                ),
            )
            .assign(analysis_type="EVALUATION_ONLY_GROUND_TRUTH_DIAGNOSTIC")
        )
    else:
        fault_diagnostics = pd.DataFrame()
    fault_diagnostics.to_csv(report_path / "fault_diagnostics.csv", index=False)
    summary = availability_summary(bundle.snapshots, tolerance=config.availability_tolerance)
    summary.update(
        {
            "mean_charge_power_availability": time_weighted_mean(
                bundle.snapshots["charge_power_availability"],
                bundle.snapshots["timestamp_utc"],
            ),
            "mean_charge_energy_availability": time_weighted_mean(
                bundle.snapshots["charge_energy_availability"],
                bundle.snapshots["timestamp_utc"],
            ),
            "mean_requested_energy_availability": time_weighted_mean(
                bundle.snapshots["requested_energy_availability"],
                bundle.snapshots["timestamp_utc"],
            ),
            "total_rack_downtime_minutes": float(rack_downtime["downtime_minutes"].sum()),
            "total_pcs_downtime_minutes": float(pcs_downtime["downtime_minutes"].sum()),
            "rack_availability_event_count": len(rack_events),
            "pcs_availability_event_count": len(pcs_events),
        }
    )
    high_technical_low_power = bundle.snapshots["technical_availability"].ge(
        0.999
    ) & bundle.snapshots["power_availability"].lt(0.999)
    high_power_low_energy = bundle.snapshots["power_availability"].ge(0.999) & bundle.snapshots[
        "discharge_energy_availability"
    ].lt(0.5)
    capability_ok_delivery_failed = (
        bundle.snapshots["requested_power_is_active"]
        & bundle.snapshots["requested_power_availability"].ge(1 - config.availability_tolerance)
        & bundle.snapshots["delivery_ratio"].lt(simulation_config.delivery_failure_ratio)
    )
    science = {
        "technical_high_power_reduced_interval_count": int(high_technical_low_power.sum()),
        "technical_high_power_reduced_time_fraction": _time_fraction(
            high_technical_low_power, bundle.snapshots["timestamp_utc"]
        ),
        "power_high_energy_constrained_interval_count": int(high_power_low_energy.sum()),
        "power_high_energy_constrained_time_fraction": _time_fraction(
            high_power_low_energy, bundle.snapshots["timestamp_utc"]
        ),
        "capability_adequate_delivery_failed_interval_count": int(
            capability_ok_delivery_failed.sum()
        ),
        "capability_adequate_delivery_failed_fraction_of_active": float(
            capability_ok_delivery_failed.sum()
            / bundle.snapshots["requested_power_is_active"].sum()
        ),
        "anomaly_relationship": anomaly_context,
    }
    limiting_counts: dict[str, int] = {}
    for factors in bundle.snapshots["limiting_factors"]:
        for factor in factors:
            limiting_counts[str(factor)] = limiting_counts.get(str(factor), 0) + 1
    output = {
        "experiment": {
            "intervals": intervals,
            "interval_minutes": simulation_config.battery.telemetry_interval_minutes,
            "seed": simulation_config.simulation_seed,
            "start_timestamp": bundle.snapshots["timestamp_utc"].min(),
            "end_timestamp": bundle.snapshots["timestamp_utc"].max(),
            "fault_event_count": len(faults),
        },
        "definitions": {
            "power_sign": "positive discharge; negative charge",
            "energy_basis": config.energy_reporting_basis,
            "available_energy_mwh_alias": "available_discharge_energy_mwh",
            "idle_request_policy": "requested availability is null/not applicable",
            "unknown_denominator_policy": (
                "unknown is not counted available; installed denominator retained"
            ),
            "future_fill": "prohibited; exact timestamp rows only",
        },
        "summary": summary,
        "scientific_questions": science,
        "limiting_factor_interval_counts": limiting_counts,
        "legacy_simulator_energy_note": (
            "Simulator SiteTelemetry.available_energy_mwh is stored DC energy above SOC minimum; "
            "Prompt 7 reports deliverable AC discharge energy and keeps the legacy field unchanged."
        ),
        "prompt5_24h_warning": "known poor out-of-time calibration; descriptive context only",
        "provenance": "DERIVED ENGINEERING ANALYTIC from SIMULATED telemetry",
    }
    _write_json(report_path / "availability_summary.json", output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Prompt 7 availability experiment")
    parser.add_argument("--reports", default="reports/availability")
    parser.add_argument("--intervals", type=int, default=2880)
    args = parser.parse_args(argv)
    result = run_availability_experiment(args.reports, intervals=args.intervals)
    summary = result["summary"]
    print(
        f"snapshots={summary['snapshot_count']} "
        f"mean_technical={summary['mean_technical_availability']:.4f} "
        f"request_support={summary['request_support_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
