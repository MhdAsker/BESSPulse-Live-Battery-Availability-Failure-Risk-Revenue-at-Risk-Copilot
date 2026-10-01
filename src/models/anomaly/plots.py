"""Optional anomaly timeline and detector-comparison plots."""

from pathlib import Path

import pandas as pd


def save_anomaly_timeline(
    intervals: pd.DataFrame,
    output_path: str | Path,
    *,
    fault_start: pd.Timestamp | None = None,
    fault_end: pd.Timestamp | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the plots extra to create anomaly figures") from exc
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(intervals["timestamp_utc"], intervals["anomaly_score"], label="Anomaly score")
    axis.plot(intervals["timestamp_utc"], intervals["threshold"], label="Threshold")
    if fault_start is not None:
        axis.axvline(
            fault_start.timestamp() / 86400,
            color="red",
            linestyle="--",
            label="Fault start (evaluation truth)",
        )
    if fault_end is not None:
        axis.axvline(
            fault_end.timestamp() / 86400,
            color="red",
            linestyle=":",
            label="Fault end (evaluation truth)",
        )
    axis.set(
        xlabel="UTC time", ylabel="Anomaly score", title="SIMULATED telemetry anomaly timeline"
    )
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
