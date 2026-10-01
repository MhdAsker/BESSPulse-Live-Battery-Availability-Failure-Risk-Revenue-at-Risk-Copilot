"""Grouped industrial rack tiles."""

from typing import Any

import streamlit as st

from dashboard.utils import finite, format_number, format_percent


def render_rack_groups(rows: list[dict[str, Any]], metric: str) -> None:
    by_id = {str(row.get("rack_id")): row for row in rows}
    for pcs_index in range(1, 5):
        pcs = f"PCS-{pcs_index:02d}"
        group = [
            by_id.get(f"{pcs}-RACK-{rack:02d}", {"rack_id": f"{pcs}-RACK-{rack:02d}"})
            for rack in range(1, 9)
        ]
        available = sum(
            bool(item.get("availability", item.get("technical_available", False))) for item in group
        )
        st.markdown(f"#### {pcs} · {available}/8 racks available")
        columns = st.columns(8)
        for column, rack in zip(columns, group, strict=True):
            value, unit = _metric_value(rack, metric)
            color = _tile_color(
                value,
                metric,
                bool(rack.get("availability", rack.get("technical_available", False))),
            )
            column.markdown(
                f'<div class="bp-rack" style="border-top:3px solid {color}"><b>R{str(rack["rack_id"])[-2:]}</b>'
                f'<br><span style="font-size:.75rem;color:#8FA1B3">{metric}</span><br>'
                f'<span style="font-size:.9rem">{_format(value, unit)}</span></div>',
                unsafe_allow_html=True,
            )


def _metric_value(row: dict[str, Any], metric: str) -> tuple[Any, str]:
    mapping = {
        "Temperature Residual": ("thermal_residual_c", "°C"),
        "Anomaly Score": ("anomaly_score", ""),
        "Availability": ("availability", "state"),
        "Voltage Spread": ("voltage_spread_v", "V"),
        "Technical State": ("technical_available", "state"),
        "SOC": ("soc", "%"),
        "RTE": ("rte", "%"),
        "Peer Temperature Deviation": ("peer_temperature_deviation_c", "°C"),
    }
    key, unit = mapping[metric]
    return row.get(key), unit


def _format(value: Any, unit: str) -> str:
    if unit == "%":
        return format_percent(value)
    if unit == "state":
        return "AVAILABLE" if value else "UNAVAILABLE"
    return format_number(value, unit)


def _tile_color(value: Any, metric: str, available: bool) -> str:
    if not available:
        return "#ED6A5A"
    number = finite(value)
    if metric == "Anomaly Score" and number is not None and number >= 0.5:
        return "#E8B04A"
    if metric == "Temperature Residual" and number is not None and abs(number) >= 3:
        return "#E8B04A"
    return "#63D48A"
