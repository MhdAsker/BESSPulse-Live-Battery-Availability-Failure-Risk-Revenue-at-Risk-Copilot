"""Idempotent persistence for deterministic availability snapshots."""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import AvailabilitySnapshotModel


def persist_availability_snapshots(session: Session, snapshots: pd.DataFrame) -> int:
    required = {
        "timestamp_utc",
        "asset_id",
        "technical_availability",
        "available_discharge_power_mw",
        "available_charge_power_mw",
    }
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"Missing availability snapshot columns: {missing}")
    written = 0
    raw_rows = snapshots.to_dict(orient="records")
    rows: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()} for row in raw_rows
    ]
    for row in rows:
        existing = session.scalar(
            select(AvailabilitySnapshotModel.id).where(
                AvailabilitySnapshotModel.timestamp_utc == row["timestamp_utc"],
                AvailabilitySnapshotModel.asset_id == str(row["asset_id"]),
            )
        )
        if existing is not None:
            continue
        session.add(
            AvailabilitySnapshotModel(
                timestamp_utc=row["timestamp_utc"],
                asset_id=str(row["asset_id"]),
                technical_availability=float(row["technical_availability"]),
                rack_technical_availability=float(row["rack_technical_availability"]),
                pcs_technical_availability=float(row["pcs_technical_availability"]),
                known_component_fraction=float(row["known_component_fraction"]),
                available_discharge_power_mw=float(row["available_discharge_power_mw"]),
                available_charge_power_mw=float(row["available_charge_power_mw"]),
                available_discharge_energy_mwh=float(row["available_discharge_energy_mwh"]),
                available_charge_energy_mwh=float(row["available_charge_energy_mwh"]),
                discharge_power_availability=float(row["discharge_power_availability"]),
                charge_power_availability=float(row["charge_power_availability"]),
                discharge_energy_availability=float(row["discharge_energy_availability"]),
                charge_energy_availability=float(row["charge_energy_availability"]),
                requested_power_mw=float(row["requested_power_mw"]),
                requested_power_availability=(
                    float(row["requested_power_availability"])
                    if pd.notna(row.get("requested_power_availability"))
                    else None
                ),
                requested_energy_availability=(
                    float(row["requested_energy_availability"])
                    if pd.notna(row.get("requested_energy_availability"))
                    else None
                ),
                sustainable_request_duration_hours=(
                    float(row["sustainable_request_duration_hours"])
                    if pd.notna(row.get("sustainable_request_duration_hours"))
                    else None
                ),
                available_racks=int(row["available_racks"]),
                unknown_racks=int(row["unknown_racks"]),
                available_pcs=int(row["available_pcs"]),
                unknown_pcs=int(row["unknown_pcs"]),
                limiting_factor=str(row["limiting_factor"]),
                limiting_factors_json=json.dumps(list(row["limiting_factors"])),
                limiting_component_ids_json=json.dumps(list(row["limiting_component_ids"])),
                data_provenance="DERIVED ENGINEERING ANALYTIC",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        written += 1
    session.commit()
    return written
