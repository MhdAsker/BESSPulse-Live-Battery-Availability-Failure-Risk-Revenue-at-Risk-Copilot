"""Gap and duplicate reporting without imputation."""

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta

from data.exceptions import EntsoeValidationError
from data.parser import parse_duration
from data.schemas import ContinuityReport, MarketObservation


def validate_series_continuity(
    observations: Iterable[MarketObservation],
    expected_interval: timedelta | None = None,
) -> ContinuityReport:
    """Report continuity for one logical series; values are never filled or changed."""

    rows = list(observations)
    if not rows:
        raise EntsoeValidationError("Cannot validate continuity of an empty series")
    interval = expected_interval
    if interval is None:
        resolutions = {row.resolution for row in rows}
        if len(resolutions) != 1:
            raise EntsoeValidationError("A continuity report requires one resolution")
        interval = parse_duration(resolutions.pop())
    if interval <= timedelta(0):
        raise EntsoeValidationError("Expected interval must be positive")
    timestamps = [row.timestamp_utc for row in rows]
    counts = Counter(timestamps)
    unique = sorted(counts)
    expected: list[datetime] = []
    current = unique[0]
    while current <= unique[-1]:
        expected.append(current)
        current += interval
    actual_set = set(unique)
    return ContinuityReport(
        expected_interval=interval,
        expected_observation_count=len(expected),
        actual_observation_count=len(rows),
        missing_timestamps=tuple(item for item in expected if item not in actual_set),
        duplicate_timestamps=tuple(item for item, count in counts.items() if count > 1),
    )
