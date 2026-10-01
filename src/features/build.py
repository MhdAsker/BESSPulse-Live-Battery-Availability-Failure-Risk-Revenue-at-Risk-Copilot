"""Reproducible Parquet feature snapshots and database-backed development CLI."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, select

from besspulse.config import SimulationConfig, load_config
from besspulse.database import BatteryTelemetryModel, MarketDataModel, RackTelemetryModel
from features.battery import build_rack_features, build_site_features
from features.market import align_market_to_battery, build_market_features
from features.peer import add_peer_features
from features.registry import FEATURE_SET_VERSION


@dataclass(frozen=True)
class SnapshotResult:
    path: Path
    metadata: dict[str, Any]
    created: bool


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered_columns = sorted(frame.columns)
    normalized = frame[ordered_columns].sort_values(
        [column for column in ["timestamp_utc", "asset_id", "rack_id"] if column in frame],
        kind="mergesort",
    )
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy().tobytes()
    schema = "|".join(f"{name}:{normalized[name].dtype}" for name in ordered_columns)
    return hashlib.sha256(schema.encode() + hashed).hexdigest()


def write_feature_snapshot(
    frames: dict[str, pd.DataFrame],
    *,
    config: SimulationConfig,
    requested_start: datetime,
    requested_end: datetime,
    output_root: str | Path = "data/processed/features",
) -> SnapshotResult:
    """Write content-addressed Parquet frames plus auditable JSON metadata."""

    if not frames:
        raise ValueError("At least one feature frame is required")
    feature_hashes = {name: _frame_hash(frame) for name, frame in sorted(frames.items())}
    source_hashes: dict[str, str] = {}
    for frame in frames.values():
        source_hashes.update(frame.attrs.get("source_data_hashes", {}))
    config_json = config.model_dump_json()
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    identity_payload = json.dumps(
        {
            "feature_set_version": FEATURE_SET_VERSION,
            "config_hash": config_hash,
            "source_hashes": source_hashes,
            "feature_hashes": feature_hashes,
            "start": requested_start.isoformat(),
            "end": requested_end.isoformat(),
        },
        sort_keys=True,
    )
    dataset_id = hashlib.sha256(identity_payload.encode()).hexdigest()[:20]
    path = Path(output_root) / f"{FEATURE_SET_VERSION}_{dataset_id}"
    metadata_path = path / "metadata.json"
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "feature_set_version": FEATURE_SET_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_start_utc": requested_start.astimezone(UTC).isoformat(),
        "input_end_utc": requested_end.astimezone(UTC).isoformat(),
        "interval_semantics": "[start, end)",
        "configuration_hash_sha256": config_hash,
        "source_data_hashes_sha256": source_hashes,
        "feature_frame_hashes_sha256": feature_hashes,
        "frames": {
            name: {"row_count": len(frame), "feature_count": len(frame.columns)}
            for name, frame in frames.items()
        },
        "data_provenance": "DERIVED",
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        return SnapshotResult(path=path, metadata=existing, created=False)
    path.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_parquet(path / f"{name}.parquet", index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SnapshotResult(path=path, metadata=metadata, created=True)


def generate_feature_frames(
    site_telemetry: pd.DataFrame,
    rack_telemetry: pd.DataFrame,
    market_observations: pd.DataFrame | None = None,
    *,
    config: SimulationConfig | None = None,
    asset_id: str = "BESS-001",
) -> dict[str, pd.DataFrame]:
    """Build validated site, rack-peer, market, and causally combined frames."""

    cfg = config or SimulationConfig()
    source_hashes = {
        "site_telemetry": _frame_hash(site_telemetry),
        "rack_telemetry": _frame_hash(rack_telemetry),
    }
    if market_observations is not None and not market_observations.empty:
        source_hashes["market_observations"] = _frame_hash(market_observations)
    site = build_site_features(site_telemetry, cfg, asset_id=asset_id)
    if "ambient_temperature_c" not in rack_telemetry and "ambient_temperature_c" in site_telemetry:
        ambient = site_telemetry[["timestamp_utc", "ambient_temperature_c"]].drop_duplicates()
        rack_telemetry = rack_telemetry.merge(
            ambient, on="timestamp_utc", how="left", validate="many_to_one"
        )
    rack = add_peer_features(build_rack_features(rack_telemetry, cfg), cfg)
    site.attrs["source_data_hashes"] = source_hashes
    rack.attrs["source_data_hashes"] = source_hashes
    frames = {"site_features": site, "rack_features": rack}
    if market_observations is not None and not market_observations.empty:
        market = build_market_features(market_observations, cfg)
        combined = align_market_to_battery(site, market, cfg)
        market.attrs["source_data_hashes"] = source_hashes
        combined.attrs["source_data_hashes"] = source_hashes
        frames["market_features"] = market
        frames["combined_site_market_features"] = combined
    return frames


def _parse_boundary(value: str) -> datetime:
    if "T" not in value:
        return datetime.combine(date.fromisoformat(value), time.min, UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Datetime boundaries must include a UTC offset")
    return parsed.astimezone(UTC)


def _load_table(engine: Any, table: Any, start: datetime, end: datetime) -> pd.DataFrame:
    statement = select(table).where(
        table.c.timestamp_utc >= start,
        table.c.timestamp_utc < end,
    )
    frame = pd.read_sql(statement, engine)
    if "timestamp_utc" in frame:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate causal BESSPulse feature snapshots")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--asset-id", default="BESS-001")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/processed/features")
    args = parser.parse_args(argv)
    start = _parse_boundary(args.start)
    end = _parse_boundary(args.end)
    if start >= end:
        parser.error("start must be before exclusive end")
    cfg = load_config(args.config)
    engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///data/besspulse.db"))
    site = _load_table(engine, BatteryTelemetryModel.__table__, start, end)
    rack = _load_table(engine, RackTelemetryModel.__table__, start, end)
    market = _load_table(engine, MarketDataModel.__table__, start, end)
    if site.empty or rack.empty:
        parser.error("requested interval has no persisted site/rack telemetry")
    frames = generate_feature_frames(site, rack, market, config=cfg, asset_id=args.asset_id)
    result = write_feature_snapshot(
        frames,
        config=cfg,
        requested_start=start,
        requested_end=end,
        output_root=args.output,
    )
    print(
        f"feature_snapshot={result.path} created={result.created} "
        f"frames={len(frames)} version={FEATURE_SET_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
