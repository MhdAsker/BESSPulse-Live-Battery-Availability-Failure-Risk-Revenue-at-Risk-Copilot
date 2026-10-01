"""Honest monitoring shell from available artifact metadata."""

import pandas as pd
import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.context import DashboardContext
from dashboard.data import ROOT
from dashboard.utils import format_timestamp


def render(ctx: DashboardContext) -> None:
    st.markdown("## Model Monitoring")
    provenance_badges("MODEL_PREDICTION", "DERIVED")
    models = []
    for name in ("expected_power", "expected_temperature", "price"):
        metadata = ctx.local.model_metadata(name)
        models.append(
            {
                "Model": metadata.get("model_name", name),
                "Version": metadata.get("model_version", "—"),
                "Feature set": metadata.get("feature_set_version", "—"),
                "Training interval": str(metadata.get("training_date_range", "—")),
                "Created": metadata.get("created_at_utc", "—"),
                "Artifact": "AVAILABLE",
            }
        )
    for horizon in (6, 12, 24):
        path = ROOT / f"artifacts/models/delivery_risk/{horizon}h/metadata.json"
        models.append(
            {
                "Model": f"Delivery risk {horizon}h",
                "Version": f"delivery_risk_{horizon}h_v1",
                "Feature set": "v1",
                "Training interval": "See artifact metadata",
                "Created": "—",
                "Artifact": "AVAILABLE" if path.exists() else "MISSING",
            }
        )
    st.dataframe(pd.DataFrame(models), hide_index=True, use_container_width=True)
    cols = st.columns(4)
    with cols[0]:
        metric_card(
            "Artifact Availability",
            f"{sum(item['Artifact'] == 'AVAILABLE' for item in models)}/{len(models)}",
            "Local trusted artifacts",
        )
    with cols[1]:
        metric_card("24h Calibration", "LIMITED", "Poor out-of-time reliability")
    with cols[2]:
        metric_card(
            "Latest Inference",
            format_timestamp(ctx.local.latest_risk().get("timestamp_utc"), ctx.timezone),
            "Persisted risk output",
        )
    with cols[3]:
        metric_card("Copilot", "NOT CONFIGURED", "Optional subsystem")
    st.warning(
        "Known limitation: 24h delivery-risk calibration is poor out of time. No real commercial-BESS model validation has occurred."
    )
    sections = st.columns(3)
    for column, title, detail in (
        (sections[0], "Input drift", "NOT YET ACTIVE"),
        (sections[1], "Residual drift", "NOT YET ACTIVE"),
        (sections[2], "Anomaly-score drift", "NOT YET ACTIVE"),
    ):
        with column:
            metric_card(title, detail, "Prompt 14 monitoring scope")
    st.caption("No drift statistic is inferred or fabricated from currently available diagnostics.")
