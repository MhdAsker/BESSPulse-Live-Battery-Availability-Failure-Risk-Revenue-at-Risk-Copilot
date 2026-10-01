"""Operations-oriented sidebar navigation."""

from typing import Any

import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.utils import format_timestamp

PAGES = (
    "Fleet Overview",
    "Live Asset",
    "Rack Heatmap",
    "ML Health",
    "Delivery Risk",
    "Market Context",
    "Revenue at Risk",
    "Alerts",
    "AI Copilot",
    "Model Monitoring",
)


def sidebar(health: dict[str, Any] | None, assets: list[dict[str, Any]]) -> str:
    with st.sidebar:
        st.markdown("## ⚡ BESSPulse")
        st.caption("OPERATIONS INTELLIGENCE")
        online = health is not None
        health_status = str(health.get("status", "offline")) if health else "offline"
        css = "bp-status-dot" if online else ""
        st.markdown(
            f'<div class="bp-status"><span class="{css}"></span>API {health_status.upper()}</div>',
            unsafe_allow_html=True,
        )
        options = [item["asset_id"] for item in assets] or ["BESS-001"]
        st.selectbox("Asset", options, key="selected_asset")
        selected = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.selectbox("Auto-refresh", ("Off", "10s", "30s", "60s"), key="refresh_interval")
        st.radio("Time display", ("UTC", "Berlin Time"), key="timezone", horizontal=True)
        st.checkbox("Allow local artifact demo fallback", key="allow_demo_fallback")
        st.divider()
        if health:
            st.caption(
                f"Last health update: {format_timestamp(health.get('timestamp_utc'), _tz())}"
            )
        provenance_badges("SIMULATED", "REAL", "DERIVED", "MODEL_PREDICTION", "COUNTERFACTUAL")
        with st.expander("Data provenance legend"):
            st.caption("REAL: externally observed market data")
            st.caption("SIMULATED: digital-twin BESS telemetry")
            st.caption("DERIVED: deterministic analytics")
            st.caption("MODEL PREDICTION: fitted-model output")
            st.caption("COUNTERFACTUAL: hypothetical benchmark")
    return selected or PAGES[0]


def product_header() -> None:
    st.markdown(
        '<div class="bp-header"><div><div class="bp-brand"><span class="bp-brand-mark">▰</span>BESSPulse</div>'
        '<div class="bp-kicker">Live battery availability, failure-risk & revenue-at-risk copilot</div></div>'
        '<div><span class="bp-badge bp-badge-simulated">SIMULATED BESS TELEMETRY</span>'
        '<span class="bp-badge bp-badge-real">REAL ENTSO-E MARKET DATA</span></div></div>',
        unsafe_allow_html=True,
    )


def _tz() -> str:
    return "UTC" if st.session_state.get("timezone") == "UTC" else "Europe/Berlin"
