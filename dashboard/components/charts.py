"""Shared Plotly constructors and formatting."""

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go

from dashboard.theme import COLORS, PLOTLY_LAYOUT


def style_figure(fig: go.Figure, title: str, *, y_title: str = "") -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title, hovermode="x unified")
    fig.update_yaxes(title_text=y_title)
    return fig


def line_chart(
    frame: Any,
    x: str,
    series: Sequence[tuple[str, str, str]],
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    for column, label, color in series:
        if column in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame[x], y=frame[column], name=label, line={"color": color, "width": 2}
                )
            )
    return style_figure(fig, title, y_title=y_title)


def probability_bar(probability: float | None, title: str) -> go.Figure:
    value = max(0.0, min(1.0, probability or 0.0)) * 100
    color = (
        COLORS["critical"]
        if value >= 65
        else COLORS["warning"]
        if value >= 35
        else COLORS["primary"]
    )
    fig = go.Figure(
        go.Bar(
            x=[value],
            y=[title],
            orientation="h",
            marker_color=color,
            text=[f"{value:.1f}%"],
            textposition="inside",
        )
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Predicted delivery-failure probability")
    fig.update_layout(height=125, showlegend=False, **PLOTLY_LAYOUT)
    return fig
