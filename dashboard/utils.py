"""Safe formatting and presentation-only status helpers."""

import math
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: Any, unit: str = "", decimals: int = 1) -> str:
    number = finite(value)
    if number is None:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{number:,.{decimals}f}{suffix}"


def format_percent(value: Any, decimals: int = 1) -> str:
    number = finite(value)
    return "—" if number is None else f"{number * 100:.{decimals}f}%"


def format_eur(value: Any, decimals: int = 0) -> str:
    number = finite(value)
    return "—" if number is None else f"€{number:,.{decimals}f}"


def format_timestamp(value: str | datetime | None, timezone: str = "UTC") -> str:
    if value is None:
        return "Unavailable"
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    target = UTC if timezone == "UTC" else ZoneInfo("Europe/Berlin")
    label = "UTC" if timezone == "UTC" else "Europe/Berlin"
    return f"{parsed.astimezone(target):%Y-%m-%d %H:%M} {label}"


def priority_presentation(level: str | None) -> tuple[str, str]:
    mapping = {
        "CRITICAL": ("✹", "critical"),
        "HIGH": ("▲", "high"),
        "MEDIUM": ("◆", "medium"),
        "LOW": ("●", "low"),
        "INFO": ("●", "info"),
    }
    return mapping.get((level or "INFO").upper(), ("●", "info"))


def operational_status(highest_priority: str | None, technical_availability: Any) -> str:
    """Presentation mapping from backend outputs; no new health score is calculated."""

    level = (highest_priority or "").upper()
    if level == "CRITICAL":
        return "CRITICAL"
    if level == "HIGH":
        return "DEGRADED"
    if level in {"MEDIUM", "LOW"}:
        return "WATCH"
    availability = finite(technical_availability)
    if availability is not None and availability < 1:
        return "DEGRADED"
    return "NORMAL"
