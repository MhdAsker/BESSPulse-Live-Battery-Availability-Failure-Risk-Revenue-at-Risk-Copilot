"""Monitoring snapshot construction and SQL persistence."""

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from besspulse.database import (
    AlertModel,
    AvailabilitySnapshotModel,
    BatteryTelemetryModel,
    MarketDataModel,
    ModelPredictionModel,
    MonitoringMetricModel,
    RevenueAtRiskModel,
    create_session_factory,
)
from monitoring.config import MonitoringConfig
from monitoring.metrics import freshness_seconds, missingness, population_stability_index
from monitoring.schemas import HealthStatus, MonitoringSnapshot


class MonitoringService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, snapshot: MonitoringSnapshot) -> MonitoringSnapshot:
        self.session.add(
            MonitoringMetricModel(
                observed_at_utc=snapshot.observed_at_utc,
                metric_scope=snapshot.metric_scope,
                subsystem=snapshot.subsystem,
                metric_name=snapshot.metric_name,
                metric_value=snapshot.metric_value,
                threshold=snapshot.threshold,
                status=snapshot.status.value,
                model_name=snapshot.model_name,
                model_version=snapshot.model_version,
                asset_id=snapshot.asset_id,
                reference_start=snapshot.reference_start,
                reference_end=snapshot.reference_end,
                current_start=snapshot.current_start,
                current_end=snapshot.current_end,
                sample_count=snapshot.sample_count,
                details_json=json.dumps(snapshot.details, sort_keys=True),
                data_provenance=snapshot.data_provenance,
            )
        )
        self.session.commit()
        return snapshot

    def list(
        self,
        *,
        scope: str | None = None,
        model_name: str | None = None,
        asset_id: str | None = None,
        status: HealthStatus | None = None,
        limit: int = 100,
    ) -> tuple[MonitoringSnapshot, ...]:
        query = select(MonitoringMetricModel).order_by(MonitoringMetricModel.observed_at_utc.desc())
        if scope:
            query = query.where(MonitoringMetricModel.metric_scope == scope)
        if model_name:
            query = query.where(MonitoringMetricModel.model_name == model_name)
        if asset_id:
            query = query.where(MonitoringMetricModel.asset_id == asset_id)
        if status:
            query = query.where(MonitoringMetricModel.status == status.value)
        records = self.session.scalars(query.limit(min(max(limit, 1), 500))).all()
        return tuple(self._snapshot(record) for record in records)

    def heartbeat(self, scope: str = "service") -> MonitoringSnapshot:
        return self.record(
            MonitoringSnapshot(
                observed_at_utc=datetime.now(UTC),
                metric_scope=scope,
                metric_name="monitoring_heartbeat",
                metric_value=1.0,
                status=HealthStatus.OK,
            )
        )

    @staticmethod
    def _snapshot(record: MonitoringMetricModel) -> MonitoringSnapshot:
        return MonitoringSnapshot(
            observed_at_utc=record.observed_at_utc,
            metric_scope=record.metric_scope,
            subsystem=record.subsystem,
            metric_name=record.metric_name,
            metric_value=record.metric_value,
            threshold=record.threshold,
            status=HealthStatus(record.status),
            model_name=record.model_name,
            model_version=record.model_version,
            asset_id=record.asset_id,
            reference_start=record.reference_start,
            reference_end=record.reference_end,
            current_start=record.current_start,
            current_end=record.current_end,
            sample_count=record.sample_count,
            details=json.loads(record.details_json),
            data_provenance=record.data_provenance,
        )


def snapshot_from_metric(
    scope: str, name: str, value: float | None, warning: float
) -> MonitoringSnapshot:
    status = (
        HealthStatus.INSUFFICIENT_DATA
        if value is None
        else (HealthStatus.WARNING if value > warning else HealthStatus.OK)
    )
    return MonitoringSnapshot(
        observed_at_utc=datetime.now(UTC),
        metric_scope=scope,
        metric_name=name,
        metric_value=value,
        threshold=warning,
        status=status,
    )


def run_monitoring(
    reference: Mapping[str, NDArray[np.float64]],
    current: Mapping[str, NDArray[np.float64]],
    *,
    asset_id: str | None = None,
    model_name: str | None = None,
    config: MonitoringConfig | None = None,
    service: MonitoringService | None = None,
) -> tuple[MonitoringSnapshot, ...]:
    """Compare curated reference/current features and optionally persist results."""

    settings = config or MonitoringConfig()
    now = datetime.now(UTC)
    snapshots: list[MonitoringSnapshot] = []
    for feature in sorted(set(reference) & set(current)):
        ref, cur = reference[feature], current[feature]
        sample_count = int(np.isfinite(cur).sum())
        psi = (
            population_stability_index(ref, cur)
            if sample_count >= settings.minimum_sample_count
            else None
        )
        status = HealthStatus.INSUFFICIENT_DATA
        if psi is not None:
            status = (
                HealthStatus.CRITICAL
                if psi >= settings.psi_critical
                else HealthStatus.WARNING
                if psi >= settings.psi_warning
                else HealthStatus.OK
            )
        snapshots.extend(
            (
                MonitoringSnapshot(
                    observed_at_utc=now,
                    metric_scope="feature_drift",
                    subsystem=feature,
                    metric_name="psi",
                    metric_value=psi,
                    threshold=settings.psi_warning,
                    status=status,
                    model_name=model_name,
                    asset_id=asset_id,
                    sample_count=sample_count,
                    details={"interpretation": "Model/data drift; not a battery fault."},
                ),
                snapshot_from_metric(
                    "data_quality",
                    f"{feature}.missing_fraction",
                    missingness(cur),
                    settings.missingness_warning,
                ).model_copy(
                    update={
                        "subsystem": feature,
                        "model_name": model_name,
                        "asset_id": asset_id,
                        "sample_count": len(cur),
                    }
                ),
            )
        )
    if service:
        for snapshot in snapshots:
            service.record(snapshot)
    return tuple(snapshots)


def run_database_monitoring(
    session: Session,
    *,
    asset_id: str | None = None,
    config: MonitoringConfig | None = None,
    persist: bool = True,
) -> tuple[MonitoringSnapshot, ...]:
    """Scheduler-safe freshness monitoring over already persisted outputs."""

    settings = config or MonitoringConfig()
    now = datetime.now(UTC)
    sources = {
        "telemetry": session.scalar(select(func.max(BatteryTelemetryModel.timestamp_utc))),
        "market": session.scalar(select(func.max(MarketDataModel.timestamp_utc))),
        "risk": session.scalar(select(func.max(ModelPredictionModel.timestamp_utc))),
        "availability": session.scalar(select(func.max(AvailabilitySnapshotModel.timestamp_utc))),
        "alerts": session.scalar(select(func.max(AlertModel.updated_at_utc))),
        "commercial": session.scalar(select(func.max(RevenueAtRiskModel.end_timestamp_utc))),
    }
    snapshots: list[MonitoringSnapshot] = []
    for subsystem, latest in sources.items():
        if latest is not None and (latest.tzinfo is None or latest.utcoffset() is None):
            latest = latest.replace(tzinfo=UTC)
        age = freshness_seconds(latest, now)
        if age is None:
            status = HealthStatus.NOT_AVAILABLE
        elif age > settings.freshness_critical.total_seconds():
            status = HealthStatus.CRITICAL
        elif age > settings.freshness_warning.total_seconds():
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.OK
        snapshots.append(
            MonitoringSnapshot(
                observed_at_utc=now,
                metric_scope="freshness",
                subsystem=subsystem,
                metric_name="age_seconds",
                metric_value=age,
                threshold=settings.freshness_warning.total_seconds(),
                status=status,
                asset_id=asset_id,
                current_end=now,
                sample_count=1 if latest is not None else 0,
                details={"latest_timestamp_utc": latest.isoformat() if latest else None},
            )
        )
    if persist:
        service = MonitoringService(session)
        for snapshot in snapshots:
            service.record(snapshot)
    return tuple(snapshots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BESSPulse monitoring scheduler hook")
    parser.add_argument("--asset")
    parser.add_argument("--model")
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    if args.persist:
        with create_session_factory()() as session:
            snapshots = run_database_monitoring(session, asset_id=args.asset, persist=True)
        print(json.dumps({"persisted": len(snapshots), "status": "complete"}))
        return
    else:
        snapshot = MonitoringSnapshot(
            observed_at_utc=datetime.now(UTC),
            metric_scope="scheduler",
            metric_name="dry_run",
            status=HealthStatus.NOT_AVAILABLE,
            asset_id=args.asset,
            model_name=args.model,
            details={"reason": "No input dataset supplied; no metrics fabricated."},
        )
    print(snapshot.model_dump_json())


if __name__ == "__main__":
    main()
