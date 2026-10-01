"""Operations-grade alert console."""

import ast
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components.alerts import alert_card
from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.context import DashboardContext
from dashboard.utils import format_eur, format_number, format_percent, format_timestamp


def render(ctx: DashboardContext) -> None:
    st.markdown("## Alerts")
    provenance_badges("DECISION_SUPPORT", "MODEL_PREDICTION", "DERIVED", "COUNTERFACTUAL")
    filter_cols = st.columns(6)
    priority = filter_cols[0].selectbox(
        "Priority", ("ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
    )
    status = filter_cols[1].selectbox("Status", ("ALL", "OPEN", "RESOLVED"))
    alert_type = filter_cols[2].selectbox(
        "Type",
        (
            "ALL",
            "DELIVERY_RISK",
            "ANOMALY",
            "POWER_DERATING",
            "ENERGY_CAPACITY_REDUCTION",
            "RACK_UNAVAILABLE",
            "PCS_UNAVAILABLE",
            "THERMAL_ISSUE",
            "POWER_TRACKING",
            "EFFICIENCY_DEGRADATION",
            "COMMERCIAL_IMPACT",
        ),
    )
    component = filter_cols[3].text_input("Component", placeholder="SITE / PCS / rack")
    search = filter_cols[4].text_input("Search")
    limit = filter_cols[5].selectbox("Rows", (25, 50, 100))
    try:
        page = ctx.client.alerts(
            asset_id=ctx.asset_id,
            status=status,
            priority_level=priority,
            alert_type=alert_type,
            component_id=component or None,
            limit=limit,
        )
        rows = [item.model_dump(mode="json") for item in page.items]
    except Exception:
        rows = ctx.local.alerts() if st.session_state.get("allow_demo_fallback") else []
        st.warning("Alerts API unavailable — using local persisted results (demo data source).")
    rows = _normalize(rows)
    if priority != "ALL":
        rows = [row for row in rows if row.get("priority_level") == priority]
    if status != "ALL":
        rows = [row for row in rows if row.get("status") == status]
    if alert_type != "ALL":
        rows = [row for row in rows if row.get("alert_type") == alert_type]
    if component:
        rows = [row for row in rows if component.upper() in str(row.get("component", "")).upper()]
    if search:
        rows = [row for row in rows if search.lower() in str(row).lower()]
    if not rows:
        st.success("No alerts match the selected filters.")
        return
    top = sorted(rows, key=lambda row: float(row.get("priority_score", 0)), reverse=True)[:3]
    for column, alert in zip(st.columns(3), top, strict=False):
        with column:
            alert_card(alert)
    table = pd.DataFrame(rows)
    columns = [
        item
        for item in (
            "priority_level",
            "component",
            "alert_type",
            "status",
            "failure_probability",
            "affected_power_mw",
            "affected_energy_mwh",
            "revenue_at_risk_eur",
            "opened_at",
            "model_confidence",
        )
        if item in table
    ]
    st.dataframe(table[columns], hide_index=True, use_container_width=True, height=330)
    choices = {
        f"{row.get('priority_level')} · {row.get('component')} · {row.get('alert_type')}": row
        for row in rows
    }
    selected_key = st.selectbox("Alert detail", tuple(choices)) or next(iter(choices))
    selected = choices[selected_key]
    st.markdown("### Priority evidence")
    cols = st.columns(5)
    evidence = (
        (
            "Priority Score",
            format_number(selected.get("priority_score"), decimals=3),
            "Bounded 0-1",
        ),
        (
            "Technical Severity",
            format_percent(selected.get("technical_severity")),
            "Engineering impact",
        ),
        (
            "Commercial Severity",
            format_percent(selected.get("commercial_severity")),
            "Counterfactual context",
        ),
        (
            "Failure Probability",
            format_percent(selected.get("failure_probability")),
            f"{selected.get('risk_horizon', selected.get('risk_horizon_hours', '—'))}h horizon",
        ),
        (
            "Model Confidence",
            format_percent(selected.get("model_confidence")),
            "Operational confidence",
        ),
    )
    for column, item in zip(cols, evidence, strict=True):
        with column:
            metric_card(*item)
    st.markdown(
        "**Impact:** "
        + " · ".join(
            (
                format_number(selected.get("affected_power_mw"), "MW"),
                format_number(selected.get("affected_energy_mwh"), "MWh"),
                format_eur(selected.get("revenue_at_risk_eur")),
            )
        )
    )
    st.caption(
        f"Limiting factor: {selected.get('limiting_factor', '—')} · Opened {format_timestamp(selected.get('opened_at'), ctx.timezone)}"
    )
    st.markdown("**Supporting signals:** " + ", ".join(_list(selected.get("supporting_signals"))))
    with st.expander("Source versions and transparent formula"):
        st.json(selected.get("source_versions", {}))
        st.caption(
            "Priority = confidence x weighted risk, capacity, commercial, and anomaly/technical components, with the documented technical override. Anomaly score is not a probability."
        )


def _normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.setdefault("component", row.get("component_id"))
        row.setdefault("risk_horizon", row.get("risk_horizon_hours"))
        row.setdefault("opened_at", row.get("opened_at_utc"))
    return rows


def _list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            return [str(item) for item in ast.literal_eval(value)]
        except (ValueError, SyntaxError):
            return [value]
    return []
