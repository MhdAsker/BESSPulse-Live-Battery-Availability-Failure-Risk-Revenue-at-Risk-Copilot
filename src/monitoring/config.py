"""Project monitoring assumptions; thresholds are configurable, not universal standards."""

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class MonitoringConfig:
    current_window: timedelta = timedelta(hours=24)
    reference_window: timedelta = timedelta(days=30)
    minimum_sample_count: int = 30
    freshness_warning: timedelta = timedelta(minutes=15)
    freshness_critical: timedelta = timedelta(hours=1)
    psi_warning: float = 0.2
    psi_critical: float = 0.4
    missingness_warning: float = 0.05
