"""BESSPulse Streamlit entry point."""

from collections.abc import Callable
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from dashboard.api_client import BESSPulseAPIClient, DashboardAPIError
from dashboard.components.navigation import product_header, sidebar
from dashboard.components.status import system_health_strip
from dashboard.config import get_dashboard_settings
from dashboard.context import DashboardContext
from dashboard.data import LocalArtifactData
from dashboard.pages import (
    ai_copilot,
    alerts,
    delivery_risk,
    fleet_overview,
    live_asset,
    market_context,
    ml_health,
    model_monitoring,
    rack_heatmap,
    revenue_at_risk,
)
from dashboard.state import initialize_state
from dashboard.theme import apply_theme

PAGE_RENDERERS: dict[str, Callable[[DashboardContext], None]] = {
    "Fleet Overview": fleet_overview.render,
    "Live Asset": live_asset.render,
    "Rack Heatmap": rack_heatmap.render,
    "ML Health": ml_health.render,
    "Delivery Risk": delivery_risk.render,
    "Market Context": market_context.render,
    "Revenue at Risk": revenue_at_risk.render,
    "Alerts": alerts.render,
    "AI Copilot": ai_copilot.render,
    "Model Monitoring": model_monitoring.render,
}


@st.cache_data(ttl=10, show_spinner=False)
def _health(base_url: str, timeout: float, retries: int) -> dict[str, Any] | None:
    try:
        return (
            BESSPulseAPIClient(base_url, timeout_seconds=timeout, retries=retries)
            .health()
            .model_dump(mode="json")
        )
    except DashboardAPIError:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _assets(base_url: str, timeout: float, retries: int) -> list[dict[str, Any]]:
    try:
        values = BESSPulseAPIClient(base_url, timeout_seconds=timeout, retries=retries).assets()
        return [value.model_dump(mode="json") for value in values]
    except DashboardAPIError:
        return []


def main() -> None:
    st.set_page_config(
        page_title="BESSPulse",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    apply_theme()
    settings = get_dashboard_settings()
    client = BESSPulseAPIClient(
        settings.besspulse_api_url,
        timeout_seconds=settings.dashboard_api_timeout_seconds,
        retries=settings.dashboard_api_retries,
    )
    health = _health(
        settings.besspulse_api_url,
        settings.dashboard_api_timeout_seconds,
        settings.dashboard_api_retries,
    )
    assets = _assets(
        settings.besspulse_api_url,
        settings.dashboard_api_timeout_seconds,
        settings.dashboard_api_retries,
    )
    selected_page = sidebar(health, assets)
    _auto_refresh()
    product_header()
    system_health_strip(health)
    if health is None:
        st.error("API OFFLINE · Start FastAPI to enable live typed responses.")
        if st.session_state.get("allow_demo_fallback"):
            st.warning(
                "DEMO / LOCAL DATA MODE enabled · using clearly labeled persisted project artifacts where available."
            )
    context = DashboardContext(
        client=client,
        local=LocalArtifactData(),
        asset_id=st.session_state["selected_asset"],
        timezone="UTC" if st.session_state["timezone"] == "UTC" else "Europe/Berlin",
    )
    PAGE_RENDERERS[selected_page](context)


def _auto_refresh() -> None:
    seconds = {"Off": 0, "10s": 10, "30s": 30, "60s": 60}.get(
        st.session_state.get("refresh_interval", "30s"), 30
    )
    if seconds:
        components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload();}}, {seconds * 1000});</script>",
            height=0,
        )


if __name__ == "__main__":
    main()
