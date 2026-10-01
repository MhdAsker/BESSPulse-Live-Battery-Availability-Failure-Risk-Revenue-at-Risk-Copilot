"""Accessible provenance and status badges."""

from html import escape

import streamlit as st

PROVENANCE_STYLES = {
    "REAL": ("real", "REAL"),
    "SIMULATED": ("simulated", "SIMULATED"),
    "DERIVED": ("derived", "DERIVED"),
    "MODEL_PREDICTION": ("model", "MODEL PREDICTION"),
    "COUNTERFACTUAL": ("counterfactual", "COUNTERFACTUAL"),
    "DECISION_SUPPORT": ("decision", "DECISION SUPPORT"),
}


def badge_html(label: str) -> str:
    style, display = PROVENANCE_STYLES.get(label.upper(), ("derived", label.upper()))
    return f'<span class="bp-badge bp-badge-{style}">{escape(display)}</span>'


def provenance_badges(*labels: str) -> None:
    st.markdown("".join(badge_html(label) for label in labels), unsafe_allow_html=True)


def calibration_badge(status: str) -> str:
    style = "counterfactual" if status.lower() == "limited" else "derived"
    return f'<span class="bp-badge bp-badge-{style}">CALIBRATION {escape(status.upper())}</span>'
