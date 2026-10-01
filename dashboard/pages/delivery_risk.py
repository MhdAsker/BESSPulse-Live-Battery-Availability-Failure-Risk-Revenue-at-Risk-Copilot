"""Delivery-risk probability, calibration, and explanation page."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.badges import calibration_badge, provenance_badges
from dashboard.components.cards import metric_card
from dashboard.components.charts import probability_bar, style_figure
from dashboard.context import DashboardContext
from dashboard.data import ROOT
from dashboard.theme import COLORS
from dashboard.utils import format_number, format_timestamp


def render(ctx: DashboardContext) -> None:
    st.markdown("## Delivery Risk")
    provenance_badges("MODEL_PREDICTION", "DERIVED")
    risk, demo = ctx.fetch(
        lambda: ctx.client.delivery_risk(ctx.asset_id), ctx.local.latest_risk, "Delivery risk"
    )
    if not risk:
        return
    cards = st.columns(3)
    calibration = {int(item["horizon_hours"]): item for item in risk.get("calibration", [])}
    for column, horizon in zip(cards, (6, 12, 24), strict=True):
        probability = risk.get(f"failure_probability_{horizon}h")
        status = calibration.get(horizon, {}).get(
            "status", "limited" if horizon == 24 else "validated"
        )
        with column:
            st.plotly_chart(
                probability_bar(probability, f"{horizon} HOURS"), use_container_width=True
            )
            metric_card(
                "Model",
                str(risk.get(f"model_version_{horizon}h", "—")),
                f"As of {format_timestamp(risk.get('timestamp_utc'), ctx.timezone)}",
            )
            st.markdown(calibration_badge(str(status)), unsafe_allow_html=True)
    st.warning(
        "24h calibration is limited due to poor out-of-time reliability. The probability remains visible but should be interpreted cautiously."
    )

    metrics = ctx.local.delivery_metrics()
    cols = st.columns(3)
    for column, horizon in zip(cols, (6, 12, 24), strict=True):
        with column:
            metric_card(
                f"{horizon}h Brier Score",
                format_number(metrics.get(str(horizon), {}).get("brier_score"), decimals=3),
                "Lower is better · offline simulated test",
            )
    selected = (
        st.selectbox("Calibration horizon", (6, 12, 24), format_func=lambda value: f"{value} hours")
        or 6
    )
    reliability_path = ROOT / f"reports/delivery_risk/reliability_{selected}h.json"
    reliability = (
        json.loads(reliability_path.read_text(encoding="utf-8"))
        if reliability_path.exists()
        else []
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name="Ideal", line={"dash": "dot", "color": COLORS["muted"]})
    )
    figure.add_trace(
        go.Scatter(
            x=[row["mean_probability"] for row in reliability],
            y=[row["observed_frequency"] for row in reliability],
            mode="lines+markers",
            name=f"{selected}h reliability",
            line={"color": COLORS["secondary"]},
        )
    )
    figure.update_xaxes(title="Mean predicted probability", range=[0, 1])
    figure.update_yaxes(title="Observed failure frequency", range=[0, 1])
    st.plotly_chart(style_figure(figure, "Reliability diagram"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        shap_path = ROOT / f"artifacts/models/delivery_risk/{selected}h/global_shap_importance.csv"
        shap = pd.read_csv(shap_path).head(10) if shap_path.exists() else pd.DataFrame()
        if not shap.empty:
            fig = go.Figure(
                go.Bar(
                    x=shap["mean_absolute_shap"],
                    y=shap["feature"].str.replace("numeric__", ""),
                    orientation="h",
                    marker_color=COLORS["secondary"],
                )
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(
                style_figure(fig, "Global feature associations", y_title="Feature"),
                use_container_width=True,
            )
            st.caption(
                "Features associated with higher/lower predicted risk; this is not a causal explanation."
            )
    with right:
        history = ctx.local.delivery_predictions(selected).tail(288)
        probability_name = f"failure_probability_{selected}h"
        fig = go.Figure(
            go.Scatter(
                x=history["timestamp_utc"],
                y=history[probability_name],
                name="Predicted risk",
                line={"color": COLORS["secondary"]},
            )
        )
        target = f"failure_within_{selected}h"
        failures = history.loc[history.get(target, 0) == 1]
        if target in history:
            fig.add_trace(
                go.Scatter(
                    x=failures["timestamp_utc"],
                    y=failures[probability_name],
                    name="Observed future failure",
                    mode="markers",
                    marker={"color": COLORS["critical"], "symbol": "x", "size": 9},
                )
            )
        st.plotly_chart(
            style_figure(fig, "Historical prediction vs outcome", y_title="Probability"),
            use_container_width=True,
        )
        st.caption(
            "SIMULATED EVALUATION GROUND TRUTH · offline evaluation only, never live operational state."
        )
    if demo:
        st.caption("Current risk cards use DEMO / LOCAL DATA MODE.")
