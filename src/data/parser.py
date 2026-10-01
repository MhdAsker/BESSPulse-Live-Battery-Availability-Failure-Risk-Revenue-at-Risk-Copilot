"""Namespace-robust ENTSO-E XML normalization."""

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from data.exceptions import (
    EntsoeAuthenticationError,
    EntsoeDataUnavailableError,
    EntsoeParseError,
    EntsoeRequestError,
    EntsoeValidationError,
)
from data.schemas import Dataset, MarketMetric, MarketObservation, SeriesType

_DURATION_PATTERN = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?$")
_PSR_METRICS: dict[str, tuple[MarketMetric, MarketMetric]] = {
    "B16": (MarketMetric.SOLAR_GENERATION, MarketMetric.SOLAR_GENERATION_FORECAST),
    "B18": (
        MarketMetric.WIND_OFFSHORE_GENERATION,
        MarketMetric.WIND_OFFSHORE_GENERATION_FORECAST,
    ),
    "B19": (
        MarketMetric.WIND_ONSHORE_GENERATION,
        MarketMetric.WIND_ONSHORE_GENERATION_FORECAST,
    ),
}


def parse_duration(value: str) -> timedelta:
    """Parse supported ISO-8601 hour/minute resolutions without approximation."""

    match = _DURATION_PATTERN.fullmatch(value)
    if match is None:
        raise EntsoeParseError(f"Unsupported ENTSO-E resolution: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    duration = timedelta(hours=hours, minutes=minutes)
    if duration <= timedelta(0):
        raise EntsoeParseError(f"Unsupported ENTSO-E resolution: {value!r}")
    return duration


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local_name(child.tag) == name]


def _first_text(element: ET.Element, path: tuple[str, ...]) -> str | None:
    current = element
    for name in path:
        matches = _children(current, name)
        if not matches:
            return None
        current = matches[0]
    return current.text.strip() if current.text else None


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        raise EntsoeParseError("ENTSO-E period is missing a start timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EntsoeParseError("ENTSO-E period has an invalid timestamp") from exc
    offset = timestamp.utcoffset()
    if offset is None:
        raise EntsoeParseError("ENTSO-E period timestamp is not timezone-aware")
    return timestamp.astimezone(UTC)


def _classify(dataset: Dataset, psr_type: str | None) -> tuple[MarketMetric, SeriesType] | None:
    if dataset is Dataset.DAY_AHEAD_PRICES:
        return MarketMetric.DAY_AHEAD_PRICE, SeriesType.DAY_AHEAD
    if dataset is Dataset.ACTUAL_LOAD:
        return MarketMetric.ACTUAL_LOAD, SeriesType.ACTUAL
    if dataset is Dataset.LOAD_FORECAST:
        return MarketMetric.LOAD_FORECAST, SeriesType.FORECAST
    if psr_type not in _PSR_METRICS:
        # Generation documents commonly include unrelated production categories.
        return None
    actual, forecast = _PSR_METRICS[psr_type]
    if dataset is Dataset.ACTUAL_GENERATION:
        return actual, SeriesType.ACTUAL
    if dataset is Dataset.GENERATION_FORECAST:
        return forecast, SeriesType.FORECAST
    raise EntsoeValidationError(f"Unsupported dataset: {dataset}")


def _raise_if_error_document(root: ET.Element) -> None:
    if "Acknowledgement" not in _local_name(root.tag):
        return
    reason = next((item.text or "" for item in _descendants(root, "text")), "")
    lowered = reason.lower()
    if "authentication" in lowered or "token" in lowered or "unauthor" in lowered:
        raise EntsoeAuthenticationError("ENTSO-E authentication failed")
    if "no matching data" in lowered or "no data" in lowered:
        raise EntsoeDataUnavailableError("ENTSO-E returned no matching data")
    raise EntsoeRequestError("ENTSO-E rejected the request")


def parse_entsoe_xml(
    content: bytes,
    dataset: Dataset,
    market_region: str,
    retrieved_at_utc: datetime,
    content_hash: str | None = None,
) -> tuple[MarketObservation, ...]:
    """Parse points using period_start + (1-based position - 1) * resolution."""

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise EntsoeParseError("Malformed ENTSO-E XML response") from exc
    _raise_if_error_document(root)
    document_id = _first_text(root, ("mRID",))
    digest = content_hash or hashlib.sha256(content).hexdigest()
    observations: dict[tuple[str, MarketMetric, datetime], MarketObservation] = {}
    time_series = _descendants(root, "TimeSeries")
    for series in time_series:
        series_id = _first_text(series, ("mRID",)) or ""
        business_type = _first_text(series, ("businessType",))
        process_type = _first_text(series, ("process.processType",)) or _first_text(
            series, ("processType",)
        )
        psr_type = _first_text(series, ("MktPSRType", "psrType"))
        classification = _classify(dataset, psr_type)
        if classification is None:
            continue
        metric, series_type = classification
        unit = (
            _first_text(series, ("price_Measure_Unit.name",))
            or _first_text(series, ("quantity_Measure_Unit.name",))
            or ("EUR/MWh" if dataset is Dataset.DAY_AHEAD_PRICES else "MW")
        )
        for period in _children(series, "Period"):
            start = _parse_timestamp(_first_text(period, ("timeInterval", "start")))
            resolution_text = _first_text(period, ("resolution",))
            if resolution_text is None:
                raise EntsoeParseError("ENTSO-E period is missing resolution")
            resolution = parse_duration(resolution_text)
            for point in _children(period, "Point"):
                position_text = _first_text(point, ("position",))
                value_text = _first_text(point, ("price.amount",)) or _first_text(
                    point, ("quantity",)
                )
                if position_text is None or value_text is None:
                    raise EntsoeParseError("ENTSO-E point is missing position or value")
                try:
                    position = int(position_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise EntsoeParseError("ENTSO-E point contains a non-numeric value") from exc
                if position < 1:
                    raise EntsoeParseError("ENTSO-E point position must be 1-based")
                timestamp = start + (position - 1) * resolution
                observation = MarketObservation(
                    timestamp_utc=timestamp,
                    market_region=market_region,
                    metric=metric,
                    value=value,
                    unit=unit,
                    series_type=series_type,
                    retrieved_at_utc=retrieved_at_utc,
                    raw_content_hash=digest,
                    document_id=document_id,
                    timeseries_mrid=series_id,
                    business_type=business_type,
                    process_type=process_type,
                    psr_type=psr_type,
                    resolution=resolution_text,
                )
                key = (series_id, metric, timestamp)
                existing = observations.get(key)
                if existing is not None and existing.value != observation.value:
                    raise EntsoeValidationError(
                        "Conflicting duplicate point within one ENTSO-E document"
                    )
                observations[key] = observation
    if not observations:
        raise EntsoeDataUnavailableError("ENTSO-E document contains no supported observations")
    return tuple(
        sorted(
            observations.values(),
            key=lambda row: (row.timestamp_utc, row.metric, row.timeseries_mrid),
        )
    )


def aggregate_wind(
    observations: Iterable[MarketObservation],
) -> tuple[MarketObservation, ...]:
    """Aggregate wind only where both onshore and offshore components exist."""

    rows = list(observations)
    groups: dict[tuple[datetime, SeriesType], dict[MarketMetric, MarketObservation]] = {}
    for row in rows:
        if row.metric in {
            MarketMetric.WIND_ONSHORE_GENERATION,
            MarketMetric.WIND_OFFSHORE_GENERATION,
            MarketMetric.WIND_ONSHORE_GENERATION_FORECAST,
            MarketMetric.WIND_OFFSHORE_GENERATION_FORECAST,
        }:
            groups.setdefault((row.timestamp_utc, row.series_type), {})[row.metric] = row
    result: list[MarketObservation] = []
    for (_timestamp, series_type), components in groups.items():
        forecast = series_type is SeriesType.FORECAST
        onshore_metric = (
            MarketMetric.WIND_ONSHORE_GENERATION_FORECAST
            if forecast
            else MarketMetric.WIND_ONSHORE_GENERATION
        )
        offshore_metric = (
            MarketMetric.WIND_OFFSHORE_GENERATION_FORECAST
            if forecast
            else MarketMetric.WIND_OFFSHORE_GENERATION
        )
        if onshore_metric not in components or offshore_metric not in components:
            continue
        onshore = components[onshore_metric]
        offshore = components[offshore_metric]
        component_hashes = sorted({onshore.raw_content_hash, offshore.raw_content_hash})
        lineage = (
            component_hashes[0]
            if len(component_hashes) == 1
            else hashlib.sha256("|".join(component_hashes).encode()).hexdigest()
        )
        result.append(
            onshore.model_copy(
                update={
                    "metric": (
                        MarketMetric.WIND_GENERATION_FORECAST
                        if forecast
                        else MarketMetric.WIND_GENERATION
                    ),
                    "value": onshore.value + offshore.value,
                    "series_type": SeriesType.DERIVED,
                    "source": "ENTSO-E Transparency Platform (derived wind aggregate)",
                    "data_provenance": "DERIVED",
                    "raw_content_hash": lineage,
                    "timeseries_mrid": "derived:wind:onshore+offshore",
                }
            )
        )
    return tuple(sorted(result, key=lambda row: row.timestamp_utc))
