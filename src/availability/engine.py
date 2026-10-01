"""Availability orchestration, exact-timestamp history, and development CLI."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from availability.config import AvailabilityConfig
from availability.pcs import calculate_pcs_availability
from availability.rack import calculate_rack_availability
from availability.schemas import AvailabilitySnapshot, PCSCapability, RackCapability
from availability.site import calculate_site_availability
from availability.storage import persist_availability_snapshots
from besspulse import BatterySimulator
from besspulse.database import Base


@dataclass(frozen=True)
class AvailabilityHistoryBundle:
    snapshots: pd.DataFrame
    rack_capabilities: pd.DataFrame
    pcs_capabilities: pd.DataFrame


def _records_by(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    records = [
        {str(key): value for key, value in row.items()} for row in frame.to_dict(orient="records")
    ]
    return {str(row[column]): row for row in records}


def _site_value(row: dict[str, Any], preferred: str, legacy: str) -> float:
    if preferred in row:
        return float(row[preferred])
    if legacy in row:
        return float(row[legacy])
    raise ValueError(f"Site telemetry is missing {preferred}/{legacy}")


def calculate_availability_snapshot(
    site_telemetry: dict[str, Any],
    pcs_telemetry: pd.DataFrame,
    rack_telemetry: pd.DataFrame,
    config: AvailabilityConfig | None = None,
) -> tuple[AvailabilitySnapshot, list[RackCapability], list[PCSCapability]]:
    cfg = config or AvailabilityConfig()
    timestamp = pd.Timestamp(site_telemetry["timestamp_utc"])
    if timestamp.tzinfo is None:
        raise ValueError("site telemetry timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    pcs_rows = _records_by(pcs_telemetry, "pcs_id") if not pcs_telemetry.empty else {}
    rack_rows = _records_by(rack_telemetry, "rack_id") if not rack_telemetry.empty else {}
    rack_capabilities: list[RackCapability] = []
    pcs_capabilities: list[PCSCapability] = []
    for pcs_index in range(1, cfg.pcs_count + 1):
        pcs_id = f"PCS-{pcs_index:02d}"
        pcs_row = pcs_rows.get(pcs_id)
        parent_available = (
            bool(pcs_row["availability"])
            if pcs_row is not None and pd.notna(pcs_row.get("availability"))
            else None
        )
        child_capabilities: list[RackCapability] = []
        for rack_index in range(1, cfg.racks_per_pcs + 1):
            rack_id = f"{pcs_id}-RACK-{rack_index:02d}"
            rack_row = rack_rows.get(
                rack_id,
                {
                    "timestamp_utc": timestamp,
                    "rack_id": rack_id,
                    "pcs_id": pcs_id,
                },
            )
            capability = calculate_rack_availability(
                rack_row,
                cfg,
                reference_timestamp=timestamp,
                parent_pcs_available=parent_available,
            )
            child_capabilities.append(capability)
            rack_capabilities.append(capability)
        pcs_capabilities.append(
            calculate_pcs_availability(
                pcs_id,
                timestamp.to_pydatetime(),
                child_capabilities,
                pcs_row,
                cfg,
            )
        )
    requested = _site_value(site_telemetry, "requested_power_mw", "site_requested_power_mw")
    actual = _site_value(site_telemetry, "actual_power_mw", "site_actual_power_mw")
    snapshot = calculate_site_availability(
        timestamp.to_pydatetime(),
        rack_capabilities,
        pcs_capabilities,
        requested,
        actual,
        cfg,
    )
    return snapshot, rack_capabilities, pcs_capabilities


def calculate_availability_bundle(
    site_telemetry: pd.DataFrame,
    pcs_telemetry: pd.DataFrame,
    rack_telemetry: pd.DataFrame,
    config: AvailabilityConfig | None = None,
) -> AvailabilityHistoryBundle:
    cfg = config or AvailabilityConfig()
    if "timestamp_utc" not in site_telemetry:
        raise ValueError("Site telemetry requires timestamp_utc")
    site = site_telemetry.copy()
    site["timestamp_utc"] = pd.to_datetime(site["timestamp_utc"], utc=True, errors="raise")
    if site["timestamp_utc"].duplicated().any():
        raise ValueError("Site telemetry must be unique by timestamp")
    pcs = pcs_telemetry.copy()
    racks = rack_telemetry.copy()
    for frame in (pcs, racks):
        if not frame.empty:
            frame["timestamp_utc"] = pd.to_datetime(
                frame["timestamp_utc"], utc=True, errors="raise"
            )
    snapshot_rows: list[dict[str, Any]] = []
    rack_rows: list[dict[str, Any]] = []
    pcs_rows: list[dict[str, Any]] = []
    raw_site_rows = site.sort_values("timestamp_utc", kind="mergesort").to_dict(orient="records")
    site_rows = [{str(key): value for key, value in row.items()} for row in raw_site_rows]
    for site_row in site_rows:
        timestamp = site_row["timestamp_utc"]
        pcs_at_time = pcs.loc[pcs["timestamp_utc"].eq(timestamp)] if not pcs.empty else pcs
        racks_at_time = (
            racks.loc[racks["timestamp_utc"].eq(timestamp)] if not racks.empty else racks
        )
        snapshot, rack_caps, pcs_caps = calculate_availability_snapshot(
            site_row, pcs_at_time, racks_at_time, cfg
        )
        snapshot_rows.append(snapshot.model_dump())
        rack_rows.extend(item.model_dump() for item in rack_caps)
        pcs_rows.extend(item.model_dump() for item in pcs_caps)
    return AvailabilityHistoryBundle(
        snapshots=pd.DataFrame(snapshot_rows),
        rack_capabilities=pd.DataFrame(rack_rows),
        pcs_capabilities=pd.DataFrame(pcs_rows),
    )


def calculate_availability_history(
    site_telemetry: pd.DataFrame,
    pcs_telemetry: pd.DataFrame,
    rack_telemetry: pd.DataFrame,
    config: AvailabilityConfig | None = None,
) -> pd.DataFrame:
    return calculate_availability_bundle(
        site_telemetry, pcs_telemetry, rack_telemetry, config
    ).snapshots


def _development_run(start: datetime, end: datetime, config: AvailabilityConfig) -> pd.DataFrame:
    interval = config.simulation.battery.telemetry_interval_minutes
    count = int((end - start).total_seconds() / (interval * 60)) + 1
    index = np.arange(count)
    requested = np.where(index % 2 == 0, 2.0, -2.0).tolist()
    simulation = BatterySimulator(config.simulation).simulate(start, requested)
    site = pd.DataFrame(row.model_dump() for row in simulation.site_telemetry)
    pcs = pd.DataFrame(row.model_dump() for row in simulation.pcs_telemetry)
    racks = pd.DataFrame(row.model_dump() for row in simulation.rack_telemetry)
    return calculate_availability_history(site, pcs, racks, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calculate deterministic BESS availability")
    parser.add_argument("--asset-id", default="BESS-001")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output")
    parser.add_argument("--database-url")
    args = parser.parse_args(argv)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("CLI start/end must be timezone-aware")
    start_dt = start.tz_convert("UTC").to_pydatetime()
    end_dt = end.tz_convert("UTC").to_pydatetime()
    if end_dt < start_dt:
        raise ValueError("end must not precede start")
    frame = _development_run(start_dt, end_dt, AvailabilityConfig(asset_id=args.asset_id))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    persisted = 0
    if args.database_url:
        database = create_engine(args.database_url)
        Base.metadata.create_all(database)
        with Session(database) as session:
            persisted = persist_availability_snapshots(session, frame)
    print(
        f"snapshots={len(frame)} mean_technical={frame.technical_availability.mean():.4f} "
        f"minimum_discharge_mw={frame.available_discharge_power_mw.min():.4f} "
        f"persisted={persisted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
