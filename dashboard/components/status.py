"""System status strip."""

from typing import Any

import streamlit as st


def system_health_strip(health: dict[str, Any] | None) -> None:
    names = ("API", "DATABASE", "TELEMETRY", "MARKET", "ML", "COMMERCIAL", "COPILOT")
    components = health.get("components", {}) if health else {}
    values = {
        "API": "ONLINE" if health else "OFFLINE",
        "DATABASE": _map(components.get("database")),
        "TELEMETRY": _map(components.get("telemetry")),
        "MARKET": _map(components.get("market_data")),
        "ML": _map(components.get("delivery_risk")),
        "COMMERCIAL": _map(components.get("commercial")),
        "COPILOT": "NOT CONFIGURED",
    }
    columns = st.columns(len(names))
    for column, name in zip(columns, names, strict=True):
        column.caption(name)
        column.markdown(f"**{values[name]}**")


def _map(value: Any) -> str:
    return {"available": "ONLINE", "no_data": "DEGRADED"}.get(str(value), "OFFLINE")
