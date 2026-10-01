"""Counterfactual commercial benchmark page."""

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.components.charts import line_chart, style_figure
from dashboard.context import DashboardContext
from dashboard.theme import COLORS
from dashboard.utils import format_eur, format_number, format_percent

DISCLAIMER = "Counterfactual historical simulation, not actual commercial P&L."


def render(ctx: DashboardContext) -> None:
    st.markdown("## Revenue at Risk")
    provenance_badges("COUNTERFACTUAL", "REAL", "DERIVED")
    result, demo = ctx.fetch(
        lambda: ctx.client.revenue_risk(ctx.asset_id),
        ctx.local.commercial_summary,
        "Revenue at Risk",
    )
    if not result:
        return
    st.markdown(f'<div class="bp-disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = (
        (
            "Healthy Net Benchmark",
            format_eur(result.get("healthy_net_revenue_eur")),
            "REFERENCE HEALTHY ASSET",
        ),
        (
            "Current Net Benchmark",
            format_eur(result.get("current_net_revenue_eur")),
            "CURRENT ESTIMATED ASSET",
        ),
        ("Revenue at Risk", format_eur(result.get("revenue_at_risk_eur")), "Healthy minus current"),
        (
            "Revenue at Risk %",
            format_percent(result.get("revenue_at_risk_fraction")),
            "Counterfactual fraction",
        ),
    )
    for column, item in zip(cols, items, strict=True):
        with column:
            metric_card(*item, accent=COLORS["warning"])
    summary = ctx.local.commercial_summary("historical")
    availability = ctx.local.latest_availability()
    healthy, current = st.columns(2)
    with healthy:
        st.markdown("### Reference Healthy Asset")
        hcols = st.columns(3)
        with hcols[0]:
            metric_card("Power", "20.0 MW", "Reference")
        with hcols[1]:
            metric_card("Rated Energy", "40.0 MWh", "32.0 MWh usable")
        with hcols[2]:
            metric_card("Efficiency", "96.0% / 96.0%", "Charge / discharge")
    with current:
        st.markdown("### Current Estimated Asset")
        ccols = st.columns(3)
        with ccols[0]:
            metric_card(
                "Available Power",
                format_number(availability.get("available_discharge_power_mw"), "MW"),
                "DERIVED",
            )
        with ccols[1]:
            metric_card(
                "Available Energy",
                format_number(availability.get("available_discharge_energy_mwh"), "MWh"),
                "DERIVED",
            )
        with ccols[2]:
            metric_card(
                "Technical Availability",
                format_percent(availability.get("technical_availability")),
                "DERIVED",
            )
    dispatch = ctx.local.commercial_dispatch("historical")
    if not dispatch.empty:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(
                line_chart(
                    dispatch,
                    "timestamp_utc",
                    [
                        ("healthy_net_dispatch_power_mw", "Healthy dispatch", COLORS["secondary"]),
                        ("current_net_dispatch_power_mw", "Current dispatch", COLORS["primary"]),
                    ],
                    "Healthy vs current dispatch",
                    "MW",
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                line_chart(
                    dispatch,
                    "timestamp_utc",
                    [("price_eur_per_mwh", "REAL ENTSO-E price", COLORS["warning"])],
                    "Market price",
                    "EUR/MWh",
                ),
                use_container_width=True,
            )
        with chart_cols[1]:
            st.plotly_chart(
                line_chart(
                    dispatch,
                    "timestamp_utc",
                    [
                        ("cumulative_healthy_revenue_eur", "Healthy", COLORS["secondary"]),
                        ("cumulative_current_revenue_eur", "Current", COLORS["primary"]),
                        ("cumulative_revenue_at_risk_eur", "Revenue at Risk", COLORS["warning"]),
                    ],
                    "Cumulative counterfactual revenue",
                    "EUR",
                ),
                use_container_width=True,
            )
    attribution = result.get("attribution") or summary.get("attribution") or {}
    labels = [
        "Power Derating",
        "Energy Capacity",
        "Efficiency",
        "Availability",
        "Interaction / Unattributed",
    ]
    keys = [
        "power_derating_impact_eur",
        "energy_capacity_impact_eur",
        "efficiency_impact_eur",
        "availability_impact_eur",
        "interaction_unattributed_eur",
    ]
    values = [attribution.get(key, 0) for key in keys]
    figure = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative"] * 5,
            connector={"line": {"color": COLORS["muted"]}},
            increasing={"marker": {"color": COLORS["warning"]}},
            decreasing={"marker": {"color": COLORS["secondary"]}},
        )
    )
    st.plotly_chart(
        style_figure(figure, "Counterfactual attribution", y_title="EUR"), use_container_width=True
    )
    st.caption("Independent one-factor restoration impacts overlap and are not uniquely causal.")
    if demo:
        st.caption("DEMO / LOCAL DATA MODE uses the persisted Prompt 9 benchmark.")
