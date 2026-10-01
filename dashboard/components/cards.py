"""Compact consistent metric cards."""

from html import escape

import streamlit as st


def metric_card(label: str, value: str, note: str = "", *, accent: str | None = None) -> None:
    border = f"border-top:2px solid {accent};" if accent else ""
    st.markdown(
        f'<div class="bp-card" style="{border}"><div class="bp-card-label">{escape(label)}</div>'
        f'<div class="bp-card-value">{escape(value)}</div><div class="bp-card-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
