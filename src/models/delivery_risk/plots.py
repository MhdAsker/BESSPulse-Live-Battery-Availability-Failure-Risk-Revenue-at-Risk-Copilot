"""Optional calibration figures, excluded from normal test execution."""

from pathlib import Path
from typing import Any

import pandas as pd

from models.delivery_risk.evaluate import reliability_bins


def save_reliability_diagram(
    actual: pd.Series,
    probability: pd.Series,
    output_path: str | Path,
    *,
    horizon_hours: int,
    bins: int = 10,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the plots extra to generate reliability figures") from exc
    entries: list[dict[str, Any]] = reliability_bins(
        actual.astype(int), probability.to_numpy(dtype=float), bins
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.plot(
        [entry["mean_probability"] for entry in entries],
        [entry["observed_frequency"] for entry in entries],
        marker="o",
        label="Delivery-risk model",
    )
    axis.set(
        xlabel="Mean MODEL_PREDICTION probability",
        ylabel="Observed failure frequency (DERIVED from SIMULATED telemetry)",
        title=f"{horizon_hours}h delivery-risk reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
