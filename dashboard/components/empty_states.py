"""Consistent empty and API error states."""

from html import escape

import streamlit as st

from dashboard.api_client import DashboardAPIError


def empty_state(title: str, detail: str) -> None:
    st.markdown(
        f'<div class="bp-empty"><b>{escape(title)}</b><br>{escape(detail)}</div>',
        unsafe_allow_html=True,
    )


def api_error(error: DashboardAPIError, context: str) -> None:
    st.error(f"{context} is unavailable ({error.category}).")
    with st.expander("Technical detail"):
        st.caption(str(error))
    if st.button("Retry", key=f"retry-{context}"):
        st.cache_data.clear()
        st.rerun()
