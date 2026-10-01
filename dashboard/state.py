"""Stable Streamlit session-state initialization."""

from typing import Any

import streamlit as st

DEFAULTS: dict[str, Any] = {
    "selected_asset": "BESS-001",
    "selected_rack": "PCS-01-RACK-01",
    "timezone": "UTC",
    "refresh_interval": "30s",
    "allow_demo_fallback": True,
    "copilot_history": [],
    "page": "Fleet Overview",
}


def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
