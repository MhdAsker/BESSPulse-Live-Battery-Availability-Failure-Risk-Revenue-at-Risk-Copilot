"""Persist the deterministic final demo without mixing operational and truth data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from availability.storage import persist_availability_snapshots
from besspulse import BatterySimulator, SimulationConfig
from besspulse.config import FeatureConfig
from besspulse.database import (
    Asset,
    BatteryTelemetryModel,
    FaultGroundTruthModel,
    PCSModel,
    PCSTelemetryModel,
    RackModel,
    RackTelemetryModel,
    create_session_factory,
)
from models.anomaly.experiment import _faults
from models.anomaly.storage import persist_anomaly_events
from models.storage import persist_delivery_risk_predictions

ASSET_ID = "BESS-001"
START = datetime(2026, 2, 5, tzinfo=UTC)
INTERVALS = 2_880


def _chunks(rows: list[dict[str, Any]], size: int = 5_000) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _insert_if_empty(
    session: Session, model: type[Any], rows: list[dict[str, Any]], expected: int
) -> int:
    current = int(session.scalar(select(func.count()).select_from(model)) or 0)
    if current == expected:
        return 0
    if current:
        raise RuntimeError(
            f"Refusing to mix final-demo rows with {current} existing rows in {model.__tablename__}"
        )
    for chunk in _chunks(rows):
        session.execute(insert(model), chunk)
    session.commit()
    return len(rows)


def seed_final_demo() -> dict[str, Any]:
    """Create and persist the documented seed-42, ten-day simulated demo."""

    simulation = SimulationConfig(
        simulation_seed=42,
        features=FeatureConfig(require_full_windows=False),
    )
    index = np.arange(INTERVALS)
    magnitude = 1.8 + 0.2 * np.sin(2 * np.pi * index / 288)
    requested = np.where(index % 2 == 0, magnitude, -magnitude / 0.9216)
    ambient = 19 + 9 * np.sin(2 * np.pi * index / 288)
    run = BatterySimulator(simulation, _faults(START)).simulate(
        START, requested.tolist(), ambient.tolist()
    )

    session_factory = create_session_factory()
    counts: dict[str, int] = {}
    with session_factory() as session:
        asset = session.scalar(select(Asset).where(Asset.asset_id == ASSET_ID))
        if asset is None:
            asset = Asset(
                asset_id=ASSET_ID,
                rated_power_mw=simulation.battery.site_rated_power_mw,
                rated_energy_mwh=simulation.battery.site_rated_energy_mwh,
                data_provenance="SIMULATED",
            )
            session.add(asset)
            session.flush()
            for pcs_index in range(1, simulation.battery.pcs_count + 1):
                pcs = PCSModel(
                    pcs_id=f"PCS-{pcs_index:02d}",
                    asset_pk=asset.id,
                    rated_power_mw=(
                        simulation.battery.site_rated_power_mw / simulation.battery.pcs_count
                    ),
                )
                session.add(pcs)
                session.flush()
                for rack_index in range(1, simulation.battery.racks_per_pcs + 1):
                    session.add(
                        RackModel(
                            rack_id=f"{pcs.pcs_id}-RACK-{rack_index:02d}",
                            pcs_pk=pcs.id,
                            rated_power_mw=(
                                simulation.battery.site_rated_power_mw
                                / simulation.battery.pcs_count
                                / simulation.battery.racks_per_pcs
                            ),
                            rated_energy_mwh=(
                                simulation.battery.site_rated_energy_mwh
                                / simulation.battery.pcs_count
                                / simulation.battery.racks_per_pcs
                            ),
                        )
                    )
            session.commit()

        site_rows = [
            {"asset_id": ASSET_ID, **row.model_dump(mode="python")} for row in run.site_telemetry
        ]
        pcs_rows = [row.model_dump(mode="python") for row in run.pcs_telemetry]
        rack_rows = [row.model_dump(mode="python") for row in run.rack_telemetry]
        truth_rows = [
            {
                **row.model_dump(mode="python"),
                "fault_type": row.fault_type.value,
            }
            for row in run.fault_ground_truth
        ]
        counts["battery_telemetry"] = _insert_if_empty(
            session, BatteryTelemetryModel, site_rows, len(site_rows)
        )
        counts["pcs_telemetry"] = _insert_if_empty(
            session, PCSTelemetryModel, pcs_rows, len(pcs_rows)
        )
        counts["rack_telemetry"] = _insert_if_empty(
            session, RackTelemetryModel, rack_rows, len(rack_rows)
        )
        counts["fault_ground_truth"] = _insert_if_empty(
            session, FaultGroundTruthModel, truth_rows, len(truth_rows)
        )

        availability = pd.read_parquet("reports/availability/availability_snapshots.parquet")
        counts["availability_snapshots"] = persist_availability_snapshots(session, availability)
        for horizon in (6, 12, 24):
            metadata = json.loads(
                Path(f"artifacts/models/delivery_risk/{horizon}h/metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            predictions = pd.read_parquet(
                f"artifacts/models/delivery_risk/{horizon}h/test_predictions.parquet"
            )
            counts[f"delivery_risk_{horizon}h"] = persist_delivery_risk_predictions(
                session,
                predictions,
                horizon_hours=horizon,
                model_name=str(metadata["model_name"]),
                model_version=str(metadata["model_version"]),
                decision_threshold=float(metadata["classification_threshold"]),
            )
        anomalies = pd.read_parquet("reports/anomaly/events_vote_ensemble.parquet")
        counts["anomaly_events"] = persist_anomaly_events(session, anomalies)

    return {
        "asset_id": ASSET_ID,
        "start_timestamp_utc": START.isoformat(),
        "end_timestamp_utc": run.site_telemetry[-1].timestamp_utc.isoformat(),
        "intervals": INTERVALS,
        "seed": 42,
        "site_rows": len(run.site_telemetry),
        "pcs_rows": len(run.pcs_telemetry),
        "rack_rows": len(run.rack_telemetry),
        "truth_rows": len(run.fault_ground_truth),
        "writes": counts,
        "battery_provenance": "SIMULATED BESS TELEMETRY",
        "ground_truth_boundary": "SIMULATED EVALUATION ONLY",
    }


if __name__ == "__main__":
    print(json.dumps(seed_final_demo(), indent=2, sort_keys=True))
