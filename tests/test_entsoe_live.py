import os
from datetime import UTC, datetime

import pytest

from data.entsoe import EntsoeClient


@pytest.mark.skipif(
    os.getenv("RUN_ENTSOE_INTEGRATION_TESTS") != "1" or not os.getenv("ENTSOE_API_TOKEN"),
    reason="live ENTSO-E test requires explicit opt-in and token",
)
def test_live_historical_day_ahead_price_request(tmp_path) -> None:
    from data.storage import RawEntsoeStore

    with EntsoeClient.from_env(raw_store=RawEntsoeStore(tmp_path)) as client:
        result = client.fetch_day_ahead_prices(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
    assert result.observations
    assert all(row.data_provenance == "REAL" for row in result.observations)
