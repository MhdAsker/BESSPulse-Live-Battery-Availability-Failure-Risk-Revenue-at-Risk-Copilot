"""Alert presentation helpers."""

from typing import Any

import streamlit as st

from dashboard.utils import format_eur, format_number, format_percent, priority_presentation


def alert_card(alert: dict[str, Any]) -> None:
    icon, css = priority_presentation(alert.get("priority_level"))
    critical = " bp-critical" if css == "critical" else ""
    st.markdown(
        f'<div class="bp-card{critical}"><div class="bp-card-label">{icon} {alert.get("priority_level", "INFO")} · '
        f'{alert.get("component", "—")}</div><div style="font-weight:700;margin:.35rem 0">'
        f'{alert.get("alert_type", "Alert").replace("_", " ")}</div><div class="bp-card-note">'
        f"Risk {format_percent(alert.get('failure_probability'))} · "
        f"{format_number(alert.get('affected_power_mw'), 'MW')} · "
        f"{format_eur(alert.get('revenue_at_risk_eur'))}</div></div>",
        unsafe_allow_html=True,
    )
