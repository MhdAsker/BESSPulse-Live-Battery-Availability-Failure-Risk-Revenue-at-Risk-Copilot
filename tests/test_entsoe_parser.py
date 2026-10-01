from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from data.exceptions import EntsoeDataUnavailableError, EntsoeParseError
from data.parser import aggregate_wind, parse_duration, parse_entsoe_xml
from data.schemas import Dataset, MarketMetric
from data.validation import validate_series_continuity

FIXTURES = Path(__file__).parent / "fixtures" / "entsoe"
RETRIEVED = datetime(2026, 1, 2, tzinfo=UTC)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PT15M", timedelta(minutes=15)),
        ("PT30M", timedelta(minutes=30)),
        ("PT60M", timedelta(minutes=60)),
    ],
)
def test_supported_durations(text, expected) -> None:
    assert parse_duration(text) == expected


def test_unsupported_resolution_is_clear() -> None:
    with pytest.raises(EntsoeParseError, match="Unsupported"):
        parse_duration("PT45S")


def test_one_based_positions_and_negative_prices_are_preserved() -> None:
    rows = parse_entsoe_xml(
        fixture("day_ahead_prices.xml"), Dataset.DAY_AHEAD_PRICES, "DE-LU", RETRIEVED
    )
    assert [row.timestamp_utc for row in rows] == [
        datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(3)
    ]
    assert rows[1].value == -12.5
    assert all(row.data_provenance == "REAL" for row in rows)


def test_multiple_generation_series_are_sorted_and_classified() -> None:
    rows = parse_entsoe_xml(
        fixture("generation_multiple.xml"),
        Dataset.ACTUAL_GENERATION,
        "DE-LU",
        RETRIEVED,
    )
    assert len(rows) == 6
    assert [row.timestamp_utc for row in rows] == sorted(row.timestamp_utc for row in rows)
    assert {row.metric for row in rows} == {
        MarketMetric.WIND_ONSHORE_GENERATION,
        MarketMetric.WIND_OFFSHORE_GENERATION,
        MarketMetric.SOLAR_GENERATION,
    }


def test_forecast_generation_remains_distinct_and_wind_aggregate_is_derived() -> None:
    rows = parse_entsoe_xml(
        fixture("generation_multiple.xml"),
        Dataset.GENERATION_FORECAST,
        "DE-LU",
        RETRIEVED,
    )
    aggregate = aggregate_wind(rows)
    assert [row.value for row in aggregate] == [15000.0, 15200.0]
    assert all(row.metric is MarketMetric.WIND_GENERATION_FORECAST for row in aggregate)
    assert all(row.data_provenance == "DERIVED" for row in aggregate)


def test_gap_report_identifies_missing_position_without_filling() -> None:
    rows = parse_entsoe_xml(fixture("actual_load_gap.xml"), Dataset.ACTUAL_LOAD, "DE-LU", RETRIEVED)
    report = validate_series_continuity(rows)
    assert report.actual_observation_count == 3
    assert report.expected_observation_count == 4
    assert report.missing_timestamps == (datetime(2026, 3, 29, 0, 30, tzinfo=UTC),)
    assert [row.value for row in rows] == [50000.0, 50100.0, 50300.0]


def test_malformed_xml_raises_controlled_error() -> None:
    with pytest.raises(EntsoeParseError, match="Malformed"):
        parse_entsoe_xml(fixture("malformed.xml"), Dataset.ACTUAL_LOAD, "DE-LU", RETRIEVED)


def test_empty_period_is_explicitly_unavailable() -> None:
    with pytest.raises(EntsoeDataUnavailableError, match="no supported observations"):
        parse_entsoe_xml(fixture("empty_period.xml"), Dataset.ACTUAL_LOAD, "DE-LU", RETRIEVED)


def test_entsoe_no_data_error_is_distinguished() -> None:
    with pytest.raises(EntsoeDataUnavailableError, match="no matching data"):
        parse_entsoe_xml(fixture("error_response.xml"), Dataset.ACTUAL_LOAD, "DE-LU", RETRIEVED)
