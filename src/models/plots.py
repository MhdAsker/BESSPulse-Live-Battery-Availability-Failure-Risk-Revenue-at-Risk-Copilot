"""Optional, reproducible diagnostics for expected-behavior experiments."""

from pathlib import Path
from typing import Any

import pandas as pd


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install BESSPulse with the 'plots' extra to create figures") from exc
    return plt


def save_regression_diagnostics(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    residual_column: str,
    output_directory: str | Path,
    label: str,
    group_column: str | None = None,
) -> list[Path]:
    """Save deterministic actual/prediction and residual plots outside normal test runs."""

    required = {"timestamp_utc", actual_column, prediction_column, residual_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing plotting columns: {missing}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    created: list[Path] = []

    specifications = [
        ("predicted_vs_actual", actual_column, prediction_column, "Observed", "Expected"),
        ("residual_vs_predicted", prediction_column, residual_column, "Expected", "Residual"),
        ("residual_over_time", "timestamp_utc", residual_column, "UTC time", "Residual"),
    ]
    for suffix, x_column, y_column, x_label, y_label in specifications:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.scatter(frame[x_column], frame[y_column], s=8, alpha=0.65)
        axis.set(xlabel=x_label, ylabel=y_label, title=f"{label}: {suffix.replace('_', ' ')}")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = output / f"{label}_{suffix}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(frame[residual_column].dropna(), bins=30)
    axis.set(xlabel="Residual", ylabel="Count", title=f"{label}: residual distribution")
    figure.tight_layout()
    path = output / f"{label}_residual_distribution.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    created.append(path)

    if group_column is not None:
        if group_column not in frame:
            raise ValueError(f"Missing plotting group column: {group_column}")
        grouped = frame.groupby(group_column, dropna=False)[residual_column].mean()
        figure, axis = plt.subplots(figsize=(8, 4.5))
        grouped.plot.bar(ax=axis)
        axis.set(
            xlabel=group_column,
            ylabel="Mean residual",
            title=f"{label}: residual by {group_column}",
        )
        figure.tight_layout()
        path = output / f"{label}_residual_by_{group_column}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)
    return created
