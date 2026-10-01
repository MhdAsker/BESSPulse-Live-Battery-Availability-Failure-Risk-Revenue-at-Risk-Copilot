"""Idempotent normalized ENTSO-E observation persistence."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from besspulse.database import MarketDataModel
from data.schemas import MarketObservation


def persist_market_observations(
    session: Session, observations: tuple[MarketObservation, ...]
) -> int:
    """Insert new source-qualified identities and retain legitimate revisions."""

    written = 0
    for row in observations:
        identity = select(MarketDataModel.id).where(
            MarketDataModel.market_region == row.market_region,
            MarketDataModel.metric == row.metric.value,
            MarketDataModel.timestamp_utc == row.timestamp_utc,
            MarketDataModel.series_type == row.series_type.value,
            MarketDataModel.timeseries_mrid == row.timeseries_mrid,
            MarketDataModel.raw_content_hash == row.raw_content_hash,
        )
        if session.scalar(identity) is not None:
            continue
        session.add(
            MarketDataModel(
                timestamp_utc=row.timestamp_utc,
                market_region=row.market_region,
                metric=row.metric.value,
                value=row.value,
                unit=row.unit,
                series_type=row.series_type.value,
                source=row.source,
                data_provenance=row.data_provenance,
                retrieved_at_utc=row.retrieved_at_utc,
                raw_content_hash=row.raw_content_hash,
                document_id=row.document_id,
                timeseries_mrid=row.timeseries_mrid,
                business_type=row.business_type,
                process_type=row.process_type,
                psr_type=row.psr_type,
                resolution=row.resolution,
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        written += 1
    session.commit()
    return written
