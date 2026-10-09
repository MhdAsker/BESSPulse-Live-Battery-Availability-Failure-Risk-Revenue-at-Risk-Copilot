"""Freshness classification without converting missing data to success."""

from datetime import datetime, timedelta
from enum import StrEnum

from monitoring.metrics import freshness_seconds


class FreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"


def classify_freshness(
    latest: datetime | None, now: datetime, stale_after: timedelta
) -> FreshnessStatus:
    age = freshness_seconds(latest, now)
    if age is None:
        return FreshnessStatus.MISSING
    return FreshnessStatus.STALE if age > stale_after.total_seconds() else FreshnessStatus.FRESH
