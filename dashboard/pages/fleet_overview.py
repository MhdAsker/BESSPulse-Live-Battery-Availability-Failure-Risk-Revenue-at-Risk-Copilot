"""Executive fleet overview."""

import streamlit as st

from dashboard.components.alerts import alert_card
from dashboard.components.badges import calibration_badge, provenance_badges
from dashboard.components.battery import battery_visual
from dashboard.components.cards import metric_card, section_title
from dashboard.components.charts import line_chart
from dashboard.context import DashboardContext
from dashboard.theme import COLORS
from dashboard.utils import format_eur, format_number, format_percent, operational_status


def render(ctx: DashboardContext) -> None:
    st.markdown("## Fleet Overview")
    st.info("BESSPulse demo uses simulated BESS telemetry and real ENTSO-E market data.")
    status, _ = ctx.fetch(lambda: ctx.client.asset_status(ctx.asset_id), lambda: {}, "Asset status")
    availability, demo_av = ctx.fetch(
        lambda: ctx.client.availability(ctx.asset_id), ctx.local.latest_availability, "Availability"
    )
    risk, demo_risk = ctx.fetch(
        lambda: ctx.client.delivery_risk(ctx.asset_id), ctx.local.latest_risk, "Delivery risk"
    )
    commercial, _ = ctx.fetch(
        lambda: ctx.client.revenue_risk(ctx.asset_id),
        ctx.local.commercial_summary,
        "Revenue at Risk",
    )
    try:
        alerts = [
            alert_item.model_dump(mode="json")
            for alert_item in ctx.client.alerts(
                asset_id=ctx.asset_id, status="OPEN", limit=20
            ).items
        ]
    except Exception:
        alerts = ctx.local.alerts() if st.session_state.get("allow_demo_fallback") else []
    status.update(
        {
            key: value
            for key, value in availability.items()
            if key not in status or status[key] is None
        }
    )
    overall = operational_status(
        status.get("highest_priority_alert"), status.get("technical_availability")
    )
    st.markdown(
        f'<div class="bp-status"><span class="bp-status-dot"></span>{overall} · {ctx.asset_id}</div>',
        unsafe_allow_html=True,
    )
    provenance_badges("SIMULATED", "DERIVED", "MODEL_PREDICTION", "COUNTERFACTUAL")

    first = st.columns(4)
    values = (
        ("Rated Power", format_number(status.get("rated_power_mw", 20), "MW"), "Nameplate"),
        ("Rated Energy", format_number(status.get("rated_energy_mwh", 40), "MWh"), "Nameplate"),
        (
            "Available Power",
            format_number(
                status.get("available_discharge_power_mw", status.get("available_power_mw")), "MW"
            ),
            "Discharge direction",
        ),
        (
            "Available Energy",
            format_number(
                status.get("available_discharge_energy_mwh", status.get("available_energy_mwh")),
                "MWh",
            ),
            "Deliverable AC energy",
        ),
    )
    for column, card_item in zip(first, values, strict=True):
        with column:
            metric_card(*card_item)
    second = st.columns(4)
    values2 = (
        ("SOC", format_percent(status.get("site_soc")), "SIMULATED telemetry"),
        ("RTE", format_percent(status.get("rte")), "Round-trip efficiency"),
        (
            "Technical Availability",
            format_percent(status.get("technical_availability")),
            "Installed denominator",
        ),
        (
            "Requested-Power Availability",
            format_percent(status.get("requested_power_availability")),
            "Current request capability",
        ),
    )
    for column, card_item in zip(second, values2, strict=True):
        with column:
            metric_card(*card_item)

    st.markdown("### Forward risk & commercial exposure")
    risk_cols = st.columns(5)
    for column, horizon in zip(risk_cols[:3], (6, 12, 24), strict=True):
        note = "Calibration limited" if horizon == 24 else "Persisted model probability"
        with column:
            metric_card(
                f"{horizon}h Delivery Risk",
                format_percent(risk.get(f"failure_probability_{horizon}h")),
                note,
                accent=COLORS["warning"] if horizon == 24 else COLORS["secondary"],
            )
            if horizon == 24:
                st.markdown(calibration_badge("limited"), unsafe_allow_html=True)
    with risk_cols[3]:
        metric_card(
            "Revenue at Risk",
            format_eur(commercial.get("revenue_at_risk_eur")),
            "Counterfactual benchmark",
            accent=COLORS["warning"],
        )
    with risk_cols[4]:
        critical = sum(
            str(alert_item.get("priority_level")).upper() == "CRITICAL" for alert_item in alerts
        )
        metric_card(
            "Critical Alerts",
            str(critical),
            f"{len(alerts)} active total",
            accent=COLORS["critical"],
        )

    left, right = st.columns([1.25, 1])
    with left:
        battery_visual(
            status.get("site_soc"),
            status.get("available_discharge_energy_mwh", status.get("available_energy_mwh")),
            status.get("available_charge_energy_mwh"),
            "IDLE",
        )
    with right:
        section_title("Top alerts", "Transparent Prompt 10 decision-support output")
        if alerts:
            for alert in sorted(
                alerts, key=lambda row: float(row.get("priority_score", 0)), reverse=True
            )[:3]:
                alert_card(alert)
        else:
            st.success("No active critical alerts.")

    history = ctx.local.availability_history().tail(288)
    if not history.empty:
        st.info("Historical trend panels use existing local persisted simulation artifacts.")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(
                line_chart(
                    history,
                    "timestamp_utc",
                    [
                        ("available_discharge_power_mw", "Discharge MW", COLORS["primary"]),
                        ("available_charge_power_mw", "Charge MW", COLORS["secondary"]),
                    ],
                    "24h directional power capability",
                    "MW",
                ),
                use_container_width=True,
            )
        with chart_cols[1]:
            st.plotly_chart(
                line_chart(
                    history,
                    "timestamp_utc",
                    [
                        ("technical_availability", "Technical", COLORS["primary"]),
                        ("discharge_energy_availability", "Energy", COLORS["warning"]),
                    ],
                    "24h availability",
                    "Fraction",
                ),
                use_container_width=True,
            )
    if demo_av or demo_risk:
        st.caption("DEMO / LOCAL DATA MODE is active for one or more analytical panels.")
