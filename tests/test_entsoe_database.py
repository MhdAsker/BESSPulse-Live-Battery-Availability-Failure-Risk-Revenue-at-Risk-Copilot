from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from besspulse.database import Base, MarketDataModel
from data.entsoe import EntsoeClient, ingest_entsoe_market_data
from data.parser import parse_entsoe_xml
from data.persistence import persist_market_observations
from data.schemas import Dataset
from data.storage import RawEntsoeStore

FIXTURES = Path(__file__).parent / "fixtures" / "entsoe"
RETRIEVED = datetime(2026, 1, 2, tzinfo=UTC)


def observations():
    return parse_entsoe_xml(
        (FIXTURES / "day_ahead_prices.xml").read_bytes(),
        Dataset.DAY_AHEAD_PRICES,
        "DE-LU",
        RETRIEVED,
    )


def test_normalized_rows_persist_with_lineage_and_are_idempotent() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        rows = observations()
        assert persist_market_observations(session, rows) == 3
        assert persist_market_observations(session, rows) == 0
        stored = session.scalars(select(MarketDataModel)).all()
    assert len(stored) == 3
    assert stored[0].market_region == "DE-LU"
    assert stored[0].data_provenance == "REAL"
    assert stored[0].raw_content_hash == rows[0].raw_content_hash
    assert stored[1].value == -12.5


def test_market_table_has_query_and_uniqueness_indexes() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    indexes = {item["name"] for item in inspector.get_indexes("market_data")}
    constraints = {item["name"] for item in inspector.get_unique_constraints("market_data")}
    assert "ix_market_region_metric_timestamp" in indexes
    assert "uq_market_observation_source" in constraints


def test_ingestion_service_archives_parses_and_persists(tmp_path) -> None:
    content = (FIXTURES / "day_ahead_prices.xml").read_bytes()

    def handler(request):
        return httpx.Response(200, content=content, request=request)

    client = EntsoeClient(
        "test-token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        raw_store=RawEntsoeStore(tmp_path / "raw"),
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    with Session(engine) as session:
        summary = ingest_entsoe_market_data(
            start,
            end,
            datasets=(Dataset.DAY_AHEAD_PRICES,),
            client=client,
            session=session,
        )
        count = len(session.scalars(select(MarketDataModel)).all())
    assert summary.datasets_succeeded == (Dataset.DAY_AHEAD_PRICES,)
    assert summary.observations_written == 3
    assert summary.raw_files_written == 1
    assert count == 3


def test_unavailable_response_is_archived_and_reported_not_succeeded(tmp_path) -> None:
    content = (FIXTURES / "error_response.xml").read_bytes()

    def handler(request):
        return httpx.Response(200, content=content, request=request)

    client = EntsoeClient(
        "test-token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        raw_store=RawEntsoeStore(tmp_path / "raw"),
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        summary = ingest_entsoe_market_data(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datasets=(Dataset.LOAD_FORECAST,),
            client=client,
            session=session,
        )
    assert summary.datasets_succeeded == ()
    assert summary.datasets_unavailable == (Dataset.LOAD_FORECAST,)
    assert summary.raw_files_written == 1
    assert summary.observations_written == 0
