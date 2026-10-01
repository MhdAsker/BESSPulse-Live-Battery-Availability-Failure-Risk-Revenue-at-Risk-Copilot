"""Expected-behavior and anomaly model inspection."""

import pandas as pd
import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.context import DashboardContext
from dashboard.utils import format_number


def render(ctx: DashboardContext) -> None:
    st.markdown("## ML Health")
    provenance_badges("MODEL_PREDICTION", "DERIVED", "SIMULATED")
    power = ctx.local.model_metadata("expected_power")
    thermal = ctx.local.model_metadata("expected_temperature")
    delivery = ctx.local.delivery_metrics()
    anomaly = ctx.local.anomaly_metadata()
    tabs = st.tabs(("Expected Power", "Expected Temperature", "Delivery Risk", "Anomaly Detection"))
    with tabs[0]:
        _model_panel(power, "MW")
        frame = ctx.local.rack_history().tail(288 * 32)
        grouped = frame.groupby("timestamp_utc", as_index=False).agg(
            actual_power_mw=("actual_power_mw", "sum"),
            requested_power_mw=("requested_power_mw", "sum"),
        )
        grouped["power_residual_mw"] = grouped["actual_power_mw"] - grouped["requested_power_mw"]
        st.bar_chart(grouped["power_residual_mw"], height=220)
        st.caption(
            "Displayed residual is persisted actual minus requested context; expected-power evaluation metrics above come from the saved model artifact."
        )
    with tabs[1]:
        _model_panel(thermal, "°C")
        selected = st.selectbox(
            "Rack diagnostic", sorted(ctx.local.rack_history()["rack_id"].unique())
        )
        columns = ctx.local.rack_history(selected).tail(288)[
            ["timestamp_utc", "temperature_mean_c", "expected_temperature_c", "thermal_residual_c"]
        ]
        st.line_chart(
            columns.set_index("timestamp_utc")[["temperature_mean_c", "expected_temperature_c"]],
            height=270,
        )
        st.bar_chart(columns.set_index("timestamp_utc")["thermal_residual_c"], height=200)
    with tabs[2]:
        rows = []
        for horizon, metrics in delivery.items():
            rows.append(
                {
                    "Horizon": f"{horizon}h",
                    "PR-AUC": metrics.get("pr_auc_average_precision"),
                    "Brier": metrics.get("brier_score"),
                    "Recall": metrics.get("recall"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.warning("The 24h model has poor out-of-time calibration and is contextual only.")
    with tabs[3]:
        comparison = pd.DataFrame(anomaly.get("detector_comparison", []))
        wanted = [
            column
            for column in (
                "detector",
                "event_detection_rate",
                "false_alerts_per_day",
                "median_detection_delay_minutes",
                "pr_auc_average_precision",
            )
            if column in comparison
        ]
        st.dataframe(comparison[wanted], hide_index=True, use_container_width=True)
        st.caption(
            "No detector is presented as a universal winner. Metrics are simulated offline evaluation results."
        )


def _model_panel(metadata: dict[str, object], unit: str) -> None:
    metrics = metadata.get("test_metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    cols = st.columns(4)
    items = (
        (
            "Model Version",
            str(metadata.get("model_version", "—")),
            str(metadata.get("model_name", "")),
        ),
        ("MAE", format_number(metrics.get("mae"), unit, 3), "Held-out test"),
        ("RMSE", format_number(metrics.get("rmse"), unit, 3), "Held-out test"),
        ("R²", format_number(metrics.get("r2"), decimals=3), "Held-out test"),
    )
    for column, item in zip(cols, items, strict=True):
        with column:
            metric_card(*item)
    st.caption(
        f"Training interval: {metadata.get('training_date_range')} · Feature set: {metadata.get('feature_set_version')}"
    )
