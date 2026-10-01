"""Idempotent alert persistence with lifecycle updates."""

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from alerts.schemas import AlertRecord
from besspulse.database import AlertModel


def persist_alert(session: Session, alert: AlertRecord) -> bool:
    existing = session.scalar(select(AlertModel).where(AlertModel.alert_key == alert.alert_key))
    values = {
        "status": alert.status.value,
        "updated_at_utc": alert.updated_at_utc,
        "resolved_at_utc": alert.resolved_at_utc,
        "priority_score": alert.priority_score,
        "priority_level": alert.priority_level,
        "failure_probability": alert.failure_probability,
        "risk_horizon_hours": alert.risk_horizon_hours,
        "model_confidence": alert.model_confidence,
        "technical_severity": alert.technical_severity,
        "commercial_severity": alert.commercial_severity,
        "affected_power_mw": alert.affected_power_mw,
        "affected_energy_mwh": alert.affected_energy_mwh,
        "revenue_at_risk_eur": alert.revenue_at_risk_eur,
        "anomaly_score": alert.anomaly_score,
        "limiting_factor": alert.limiting_factor,
        "supporting_signals_json": json.dumps(alert.supporting_signals),
        "source_versions_json": json.dumps(alert.source_versions, sort_keys=True),
        "data_provenance": alert.data_provenance,
    }
    if existing is not None:
        for name, value in values.items():
            setattr(existing, name, value)
        session.commit()
        return False
    session.add(
        AlertModel(
            alert_key=alert.alert_key,
            asset_id=alert.asset_id,
            component_id=alert.component_id,
            component_type=alert.component_type,
            alert_type=alert.alert_type.value,
            opened_at_utc=alert.opened_at_utc,
            created_at=datetime.now(UTC),
            **values,
        )
    )
    session.commit()
    return True
