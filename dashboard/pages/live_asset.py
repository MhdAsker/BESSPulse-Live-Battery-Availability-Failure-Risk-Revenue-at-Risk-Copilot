"""Live asset operations page."""

import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.battery import battery_visual, power_flow
from dashboard.components.cards import metric_card
from dashboard.components.charts import line_chart
from dashboard.context import DashboardContext
from dashboard.theme import COLORS
from dashboard.utils import format_number, format_percent


def render(ctx: DashboardContext) -> None:
    st.markdown("## Live Asset")
    provenance_badges("SIMULATED", "DERIVED")
    frame = ctx.local.rack_history()
    availability = ctx.local.latest_availability()
    if frame.empty:
        st.info("No persisted simulation telemetry is available.")
        return
    grouped = (
        frame.groupby("timestamp_utc", as_index=False)
        .agg(
            requested_power_mw=("requested_power_mw", "sum"),
            actual_power_mw=("actual_power_mw", "sum"),
            soc=("soc", "mean"),
            temperature_mean_c=("temperature_mean_c", "mean"),
            ambient_temperature_c=("ambient_temperature_c", "mean"),
            rte=("rte", "mean"),
            alarm_count=("alarm_code", lambda values: values.notna().sum()),
        )
        .tail(288)
    )
    latest = grouped.iloc[-1]
    mode = (
        "DISCHARGING"
        if latest.actual_power_mw > 0.01
        else "CHARGING"
        if latest.actual_power_mw < -0.01
        else "IDLE"
    )
    left, right = st.columns([1, 1.5])
    with left:
        battery_visual(
            float(latest.soc),
            availability.get("available_discharge_energy_mwh"),
            availability.get("available_charge_energy_mwh"),
            mode,
        )
        power_flow(mode)
    with right:
        cols = st.columns(3)
        items = (
            (
                "Requested Power",
                format_number(latest.requested_power_mw, "MW"),
                "Positive = discharge",
            ),
            ("Actual Power", format_number(latest.actual_power_mw, "MW"), mode),
            (
                "Delivery Ratio",
                format_percent(
                    abs(latest.actual_power_mw / latest.requested_power_mw)
                    if latest.requested_power_mw
                    else None
                ),
                "Observed / requested",
            ),
            (
                "Available Power",
                format_number(availability.get("available_discharge_power_mw"), "MW"),
                "Discharge direction",
            ),
            ("Available Racks", str(availability.get("available_racks", "—")), "of 32 installed"),
            ("Available PCS", str(availability.get("available_pcs", "—")), "of 4 installed"),
        )
        for index, item in enumerate(items):
            with cols[index % 3]:
                metric_card(*item)
    st.caption("Power sign convention: positive = discharge to grid; negative = charge from grid.")
    charts = st.columns(2)
    with charts[0]:
        st.plotly_chart(
            line_chart(
                grouped,
                "timestamp_utc",
                [
                    ("requested_power_mw", "Requested", COLORS["secondary"]),
                    ("actual_power_mw", "Actual", COLORS["primary"]),
                ],
                "Requested vs actual power",
                "MW",
            ),
            use_container_width=True,
        )
        soc_fig = line_chart(
            grouped,
            "timestamp_utc",
            [("soc", "SOC", COLORS["primary"])],
            "State of charge",
            "Fraction",
        )
        soc_fig.add_hline(y=0.1, line_dash="dot", annotation_text="SOC min")
        soc_fig.add_hline(y=0.9, line_dash="dot", annotation_text="SOC max")
        st.plotly_chart(soc_fig, use_container_width=True)
        st.plotly_chart(
            line_chart(
                grouped,
                "timestamp_utc",
                [("rte", "RTE", COLORS["warning"])],
                "Round-trip efficiency",
                "Fraction",
            ),
            use_container_width=True,
        )
    with charts[1]:
        st.plotly_chart(
            line_chart(
                grouped,
                "timestamp_utc",
                [
                    ("temperature_mean_c", "Rack mean", COLORS["warning"]),
                    ("ambient_temperature_c", "Ambient", COLORS["secondary"]),
                ],
                "Temperature",
                "°C",
            ),
            use_container_width=True,
        )
        av = ctx.local.availability_history().tail(288)
        st.plotly_chart(
            line_chart(
                av,
                "timestamp_utc",
                [("available_discharge_power_mw", "Available power", COLORS["primary"])],
                "Available discharge power",
                "MW",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                av,
                "timestamp_utc",
                [
                    ("available_discharge_energy_mwh", "Available energy", COLORS["primary"]),
                    ("available_charge_energy_mwh", "Charge headroom", COLORS["secondary"]),
                ],
                "Directional energy capability",
                "MWh",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                grouped,
                "timestamp_utc",
                [("alarm_count", "Active alarms", COLORS["critical"])],
                "Alarm count",
                "Count",
            ),
            use_container_width=True,
        )
    st.caption(
        "Historical panels: DEMO / LOCAL DATA MODE · existing persisted simulated telemetry."
    )
