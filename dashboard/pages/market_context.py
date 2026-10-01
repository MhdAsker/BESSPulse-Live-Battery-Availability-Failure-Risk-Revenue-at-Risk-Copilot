"""DE-LU observed and forecast market context."""

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.badges import provenance_badges
from dashboard.components.cards import metric_card
from dashboard.components.charts import style_figure
from dashboard.context import DashboardContext
from dashboard.theme import COLORS
from dashboard.utils import format_number, format_percent, format_timestamp


def render(ctx: DashboardContext) -> None:
    st.markdown("## Market Context · DE-LU")
    provenance_badges("REAL", "MODEL_PREDICTION", "DERIVED")
    try:
        latest = ctx.client.market_latest().model_dump(mode="json")
    except Exception:
        latest = {}
    observed = latest.get("observed") or {}
    forecast_latest = latest.get("forecast") or {}
    derived = latest.get("derived") or {}
    cols = st.columns(5)
    items = (
        (
            "Day-Ahead Price",
            format_number(observed.get("day_ahead_price_eur_per_mwh"), "EUR/MWh", 2),
            "REAL ENTSO-E",
        ),
        ("Load", format_number(observed.get("load_mw"), "MW"), "REAL when available"),
        (
            "Renewables",
            format_number(observed.get("renewable_generation_mw"), "MW"),
            "Wind + solar/other",
        ),
        ("Residual Load", format_number(derived.get("residual_load_mw"), "MW"), "DERIVED"),
        ("Renewable Share", format_percent(derived.get("renewable_share")), "DERIVED"),
    )
    for column, item in zip(cols, items, strict=True):
        with column:
            metric_card(*item)
    history = ctx.local.market_history()
    forecast = ctx.local.price_forecast()
    figure = go.Figure()
    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["timestamp_utc"],
                y=history["price_eur_per_mwh"],
                name="Observed day-ahead",
                line={"color": COLORS["secondary"]},
            )
        )
        negative = history.loc[history["price_eur_per_mwh"] < 0]
        figure.add_trace(
            go.Scatter(
                x=negative["timestamp_utc"],
                y=negative["price_eur_per_mwh"],
                name="Negative price",
                mode="markers",
                marker={"color": COLORS["critical"]},
            )
        )
    if not forecast.empty:
        figure.add_trace(
            go.Scatter(
                x=forecast["target_timestamp_utc"],
                y=forecast["predicted_price_eur_per_mwh"],
                name="Point forecast",
                line={"color": COLORS["warning"], "dash": "dash"},
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    st.plotly_chart(
        style_figure(figure, "Day-ahead prices and forecast", y_title="EUR/MWh"),
        use_container_width=True,
    )
    st.caption(
        "Observed series: REAL ENTSO-E. Dashed series: MODEL PREDICTION. Negative prices remain visible."
    )
    generation_cols = st.columns(2)
    with generation_cols[0]:
        values = {
            key: observed.get(key)
            for key in (
                "load_mw",
                "wind_generation_mw",
                "solar_generation_mw",
                "renewable_generation_mw",
            )
        }
        available = {key: value for key, value in values.items() if value is not None}
        if available:
            st.bar_chart(available)
        else:
            st.info(
                "Load, wind, and solar are not present in the latest persisted market response."
            )
    with generation_cols[1]:
        metadata = ctx.local.model_metadata("price")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            metric_card(
                "Forecast Model",
                str(metadata.get("model_name", "—")),
                str(metadata.get("model_version", "")),
            )
        with metric_cols[1]:
            metric_card(
                "Forecast Horizon",
                format_number(metadata.get("forecast_horizon_hours"), "hours", 0),
                f"Origin {format_timestamp(forecast_latest.get('forecast_origin_utc'), ctx.timezone)}",
            )
        test = metadata.get("test_metrics", {})
        if isinstance(test, dict):
            st.caption(f"Saved test MAE: {format_number(test.get('mae'), 'EUR/MWh', 2)}")
    st.caption(
        "Historical chart panels use persisted local artifacts; this page is market context, not a trading interface."
    )
