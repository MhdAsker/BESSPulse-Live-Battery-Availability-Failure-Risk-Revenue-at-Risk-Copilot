from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from data.entsoe import EntsoeClient
from data.exceptions import EntsoeValidationError

BERLIN = ZoneInfo("Europe/Berlin")


def test_naive_boundary_is_rejected() -> None:
    with pytest.raises(EntsoeValidationError, match="timezone-aware"):
        EntsoeClient._validate_interval(datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC))


def test_spring_dst_conversion_uses_zone_rules() -> None:
    before = datetime(2026, 3, 29, 1, 30, tzinfo=BERLIN).astimezone(UTC)
    after = datetime(2026, 3, 29, 3, 30, tzinfo=BERLIN).astimezone(UTC)
    assert before == datetime(2026, 3, 29, 0, 30, tzinfo=UTC)
    assert after == datetime(2026, 3, 29, 1, 30, tzinfo=UTC)
    assert (after - before).total_seconds() == 3600


def test_autumn_dst_fold_maps_repeated_hour_to_distinct_utc_times() -> None:
    first = datetime(2026, 10, 25, 2, 30, tzinfo=BERLIN, fold=0).astimezone(UTC)
    second = datetime(2026, 10, 25, 2, 30, tzinfo=BERLIN, fold=1).astimezone(UTC)
    assert first == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    assert second == datetime(2026, 10, 25, 1, 30, tzinfo=UTC)
    assert first != second


def test_client_normalizes_non_utc_aware_boundaries() -> None:
    start, end = EntsoeClient._validate_interval(
        datetime(2026, 10, 25, 0, 0, tzinfo=BERLIN),
        datetime(2026, 10, 26, 0, 0, tzinfo=BERLIN),
    )
    assert start.utcoffset().total_seconds() == 0
    assert end.utcoffset().total_seconds() == 0
    assert (end - start).total_seconds() == 25 * 3600
