"""Historical simulated alert replay and machine-generated diagnostics."""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alerts.config import AlertConfig
from alerts.schemas import AlertInput, AlertStatus
from alerts.service import replay_alerts
from alerts.storage import persist_alert
from besspulse.database import create_schema


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def run_alert_experiment(
    *,
    database_url: str = "sqlite:///data/besspulse.db",
    report_dir: str | Path = "reports/alerts",
) -> dict[str, Any]:
    output = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    availability = pd.read_parquet("reports/availability/availability_snapshots.parquet")
    availability["timestamp_utc"] = pd.to_datetime(availability["timestamp_utc"], utc=True)
    for horizon in (6, 12, 24):
        risk = pd.read_parquet(
            f"artifacts/models/delivery_risk/{horizon}h/test_predictions.parquet"
        )
        risk = risk[["timestamp_utc", f"failure_probability_{horizon}h"]]
        availability = availability.merge(risk, on="timestamp_utc", how="left")
    anomaly = pd.read_parquet("reports/anomaly/site_anomaly_summary.parquet")
    availability = availability.merge(
        anomaly[["timestamp_utc", "max_component_anomaly_score", "active_anomaly_count"]],
        on="timestamp_utc",
        how="left",
    )
    commercial = json.loads(
        Path("reports/commercial/historical_benchmark_summary.json").read_text()
    )
    commercial_day = pd.Timestamp("2026-02-14", tz="UTC")
    inputs: list[AlertInput] = []
    for raw in availability.to_dict(orient="records"):
        timestamp = pd.Timestamp(raw["timestamp_utc"])
        usable = min(
            32.0,
            float(raw["available_discharge_energy_mwh"]) / 0.96
            + float(raw["available_charge_energy_mwh"]) * 0.96,
        )
        has_commercial_context = commercial_day <= timestamp < commercial_day + pd.Timedelta("24h")
        present_risks = sum(pd.notna(raw.get(f"failure_probability_{h}h")) for h in (6, 12, 24))
        inputs.append(
            AlertInput(
                timestamp_utc=timestamp.to_pydatetime(),
                available_discharge_power_mw=float(raw["available_discharge_power_mw"]),
                available_usable_energy_mwh=usable,
                technical_availability=float(raw["technical_availability"]),
                unavailable_racks=int(raw["total_racks"] - raw["available_racks"]),
                unavailable_pcs=int(raw["total_pcs"] - raw["available_pcs"]),
                limiting_factor=str(raw["limiting_factor"]),
                failure_probability_6h=(
                    float(raw["failure_probability_6h"])
                    if pd.notna(raw.get("failure_probability_6h"))
                    else None
                ),
                failure_probability_12h=(
                    float(raw["failure_probability_12h"])
                    if pd.notna(raw.get("failure_probability_12h"))
                    else None
                ),
                failure_probability_24h=(
                    float(raw["failure_probability_24h"])
                    if pd.notna(raw.get("failure_probability_24h"))
                    else None
                ),
                anomaly_score=(
                    float(raw["max_component_anomaly_score"])
                    if pd.notna(raw.get("max_component_anomaly_score"))
                    else None
                ),
                detector_count=(
                    int(raw["active_anomaly_count"])
                    if pd.notna(raw.get("active_anomaly_count"))
                    else 0
                ),
                feature_completeness=0.6 + 0.1 * present_risks,
                revenue_at_risk_eur=(
                    float(commercial["revenue_at_risk_eur"]) if has_commercial_context else 0
                ),
                revenue_at_risk_fraction=(
                    float(commercial["revenue_at_risk_fraction"]) if has_commercial_context else 0
                ),
                market_mode=(commercial["market_mode"] if has_commercial_context else None),
                price_source=(commercial["price_source"] if has_commercial_context else None),
                benchmark_disclaimer=(commercial["disclaimer"] if has_commercial_context else None),
                source_versions={
                    "delivery_risk": "6h/12h/24h_v1",
                    "anomaly": "vote_ensemble_v1",
                    "availability": "availability_v1",
                    "commercial": "commercial_dispatch_v1",
                    "feature_set": "v1",
                },
            )
        )
    records = replay_alerts(inputs, AlertConfig())
    history = pd.DataFrame([record.model_dump(mode="json") for record in records])
    history.to_csv(output / "alert_history.csv", index=False)
    ordered = history.sort_values("updated_at_utc")
    latest = ordered.drop_duplicates("alert_key", keep="last").set_index("alert_key")
    peak_indices = history.groupby("alert_key")["priority_score"].idxmax()
    unique = history.loc[peak_indices].set_index("alert_key")
    unique["status"] = latest["status"]
    unique["resolved_at_utc"] = latest["resolved_at_utc"]
    unique["updated_at_utc"] = latest["updated_at_utc"]
    unique = unique.reset_index()
    distribution = (
        unique.groupby(["priority_level", "alert_type"], dropna=False)
        .size()
        .reset_index(name="alert_count")
    )
    distribution.to_csv(output / "priority_distribution.csv", index=False)
    ranking = unique[
        ["alert_key", "technical_severity", "commercial_severity", "priority_score"]
    ].copy()
    ranking["technical_rank"] = ranking["technical_severity"].rank(ascending=False)
    ranking["priority_rank"] = ranking["priority_score"].rank(ascending=False)
    ranking.to_csv(output / "commercial_reordering.csv", index=False)
    rank_correlation = (
        float(cast(Any, ranking[["technical_rank", "priority_rank"]].corr().iloc[0, 1]))
        if len(ranking) > 1
        else None
    )

    # Ground truth enters only this post-generation evaluation block.
    truth = pd.read_parquet("reports/anomaly/evaluation_truth_cache_2880_seed42.parquet")
    starts = truth.dropna(subset=["fault_id"]).groupby("fault_id")["timestamp_utc"].min()
    opened = pd.to_datetime(unique["opened_at_utc"], utc=True)
    lead_rows = []
    for fault_id, fault_start in starts.items():
        prior = opened[(opened <= fault_start) & (opened >= fault_start - pd.Timedelta("24h"))]
        lead_rows.append(
            {
                "fault_id": fault_id,
                "fault_start_utc": fault_start,
                "prior_alert": not prior.empty,
                "lead_time_minutes": float((fault_start - prior.min()).total_seconds() / 60)
                if not prior.empty
                else None,
            }
        )
    lead = pd.DataFrame(lead_rows)
    lead.to_csv(output / "lead_time_analysis.csv", index=False)
    duration_days = (
        availability["timestamp_utc"].max() - availability["timestamp_utc"].min()
    ).total_seconds() / 86400 + 5 / 1440
    summary = {
        "total_alert_episodes": len(unique),
        "alerts_by_type": unique["alert_type"].value_counts().to_dict(),
        "alerts_by_priority": unique["priority_level"].value_counts().to_dict(),
        "alerts_per_day": len(unique) / duration_days,
        "open_count": int((unique["status"] == AlertStatus.OPEN.value).sum()),
        "resolved_count": int((unique["status"] == AlertStatus.RESOLVED.value).sum()),
        "median_affected_power_mw": float(unique["affected_power_mw"].median()),
        "median_affected_energy_mwh": float(unique["affected_energy_mwh"].median()),
        "median_revenue_at_risk_eur": float(unique["revenue_at_risk_eur"].median()),
        "faults_with_prior_alert_fraction": float(lead["prior_alert"].mean())
        if len(lead)
        else None,
        "median_lead_time_minutes": float(lead["lead_time_minutes"].median())
        if lead["lead_time_minutes"].notna().any()
        else None,
        "technical_priority_rank_correlation": rank_correlation,
        "ground_truth_usage": "EVALUATION ONLY; not present in AlertInput or generation",
        "risk_24h_limitation": "reliability weight 0.25 due poor out-of-time calibration",
    }
    _json(output / "alert_summary.json", summary)
    _json(
        output / "alert_event_metrics.json",
        {"fault_count": len(lead), "faults_with_prior_alert": int(lead["prior_alert"].sum())},
    )
    create_schema(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        new_records = sum(persist_alert(session, record) for record in records)
    metadata = {
        **summary,
        "history_rows": len(history),
        "new_database_records": new_records,
        "config": AlertConfig().model_dump(mode="json"),
    }
    _json(output / "experiment_metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_alert_experiment(), indent=2, default=str))
