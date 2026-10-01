"""Grouped 32-rack heatmap and detail diagnostics."""

import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.components.charts import line_chart
from dashboard.components.rack_heatmap import render_rack_groups
from dashboard.context import DashboardContext
from dashboard.theme import COLORS
from dashboard.utils import format_number, format_percent

METRICS = (
    "Temperature Residual",
    "Anomaly Score",
    "Availability",
    "Voltage Spread",
    "Technical State",
    "SOC",
    "RTE",
    "Peer Temperature Deviation",
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## Rack Heatmap")
    provenance_badges("SIMULATED", "DERIVED", "MODEL_PREDICTION")
    rows = ctx.local.latest_racks()
    if len(rows) != 32:
        st.warning(f"Expected 32 rack records; {len(rows)} persisted rack records are available.")
    metric = st.selectbox("Display metric", METRICS) or METRICS[0]
    render_rack_groups(rows, metric)
    rack_ids = [f"PCS-{pcs:02d}-RACK-{rack:02d}" for pcs in range(1, 5) for rack in range(1, 9)]
    selected = st.selectbox("Selected rack", rack_ids, key="selected_rack")
    row = next((item for item in rows if item.get("rack_id") == selected), {})
    st.markdown(f"### {selected} diagnostic")
    cols = st.columns(7)
    items = (
        ("SOC", format_percent(row.get("soc")), "State of charge"),
        ("Temperature", format_number(row.get("temperature_mean_c"), "°C"), "Rack mean"),
        (
            "Thermal Residual",
            format_number(row.get("thermal_residual_c"), "°C"),
            "Actual - expected",
        ),
        ("Voltage Spread", format_number(row.get("voltage_spread_v"), "V"), "Observable spread"),
        ("RTE", format_percent(row.get("rte")), "Efficiency"),
        (
            "Availability",
            "AVAILABLE" if row.get("availability") else "UNAVAILABLE",
            "Technical state",
        ),
        ("Anomaly", format_number(row.get("anomaly_score"), decimals=2), "Score, not probability"),
    )
    for column, item in zip(cols, items, strict=True):
        with column:
            metric_card(*item)
    history = ctx.local.rack_history(selected).tail(288)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            line_chart(
                history,
                "timestamp_utc",
                [
                    ("temperature_mean_c", "Actual", COLORS["warning"]),
                    ("expected_temperature_c", "Expected", COLORS["secondary"]),
                ],
                "Rack temperature history",
                "°C",
            ),
            use_container_width=True,
        )
    with chart_cols[1]:
        st.plotly_chart(
            line_chart(
                history,
                "timestamp_utc",
                [
                    ("requested_power_mw", "Requested", COLORS["secondary"]),
                    ("actual_power_mw", "Actual", COLORS["primary"]),
                ],
                "Recent power tracking",
                "MW",
            ),
            use_container_width=True,
        )
    alarms = history.loc[history["alarm_code"].notna(), ["timestamp_utc", "alarm_code"]].tail(10)
    if alarms.empty:
        st.success("No recent rack alarms.")
    else:
        st.dataframe(alarms, hide_index=True, use_container_width=True)
    st.caption(
        "DEMO / LOCAL DATA MODE · physical grouping is 4 PCS x 8 racks; no synthetic health score is shown."
    )
