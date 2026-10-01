"""Idempotent anomaly-event persistence."""

import json
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import AnomalyEventModel


def persist_anomaly_events(session: Session, events: pd.DataFrame) -> int:
    written = 0
    for row in events.to_dict(orient="records"):
        existing = session.scalar(
            select(AnomalyEventModel.id).where(
                AnomalyEventModel.anomaly_event_id == row["anomaly_event_id"]
            )
        )
        if existing is not None:
            continue
        session.add(
            AnomalyEventModel(
                anomaly_event_id=row["anomaly_event_id"],
                component_id=row["component_id"],
                component_type=row["component_type"],
                detector_name=row["detector_name"],
                detector_version=row["detector_version"],
                start_timestamp=row["start_timestamp"],
                end_timestamp=row["end_timestamp"],
                peak_score=float(row["peak_score"]),
                mean_score=float(row["mean_score"]),
                supporting_signals_json=json.dumps(row["supporting_signals"]),
                feature_set_version="v1",
                data_provenance="DERIVED",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        written += 1
    session.commit()
    return written
