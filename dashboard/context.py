"""Shared dashboard runtime context and explicit fallback routing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st
from pydantic import BaseModel

from dashboard.api_client import BESSPulseAPIClient, DashboardAPIError
from dashboard.data import LocalArtifactData


@dataclass(frozen=True)
class DashboardContext:
    client: BESSPulseAPIClient
    local: LocalArtifactData
    asset_id: str
    timezone: str

    def fetch(
        self,
        api_call: Callable[[], BaseModel],
        demo_call: Callable[[], dict[str, Any]],
        label: str,
    ) -> tuple[dict[str, Any], bool]:
        try:
            return api_call().model_dump(mode="json"), False
        except DashboardAPIError as exc:
            if st.session_state.get("allow_demo_fallback", True):
                value = demo_call()
                if value:
                    st.warning(
                        f"{label}: API unavailable — using local persisted results (demo data source)."
                    )
                    return value, True
            st.error(f"{label} unavailable ({exc.category}).")
            with st.expander("Technical detail"):
                st.caption(str(exc))
            return {}, False
