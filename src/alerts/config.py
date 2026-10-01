"""Transparent alert-priority and lifecycle configuration."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    default_formula: str = "priority_v2_weighted"
    risk_weight: float = Field(default=0.35, ge=0)
    capacity_weight: float = Field(default=0.30, ge=0)
    commercial_weight: float = Field(default=0.25, ge=0)
    anomaly_weight: float = Field(default=0.10, ge=0)
    risk_reliability_6h: float = Field(default=1.0, ge=0, le=1)
    risk_reliability_12h: float = Field(default=0.85, ge=0, le=1)
    risk_reliability_24h: float = Field(default=0.25, ge=0, le=1)
    reference_power_mw: float = Field(default=20.0, gt=0)
    reference_usable_energy_mwh: float = Field(default=32.0, gt=0)
    open_score_threshold: float = Field(default=0.30, ge=0, le=1)
    close_score_threshold: float = Field(default=0.20, ge=0, le=1)
    open_persistence_intervals: int = Field(default=2, ge=1)
    recovery_persistence_intervals: int = Field(default=3, ge=1)
    merge_window_minutes: int = Field(default=30, ge=0)
    cooldown_minutes: int = Field(default=60, ge=0)
    technical_override_threshold: float = Field(default=0.75, ge=0, le=1)
    technical_override_floor: float = Field(default=0.65, ge=0, le=1)
    info_threshold: float = Field(default=0.0, ge=0, le=1)
    low_threshold: float = Field(default=0.20, ge=0, le=1)
    medium_threshold: float = Field(default=0.40, ge=0, le=1)
    high_threshold: float = Field(default=0.65, ge=0, le=1)
    critical_threshold: float = Field(default=0.85, ge=0, le=1)
    engine_version: str = "alert_engine_v1"

    @model_validator(mode="after")
    def validate_policy(self) -> "AlertConfig":
        weights = (
            self.risk_weight + self.capacity_weight + self.commercial_weight + self.anomaly_weight
        )
        if not np.isclose(weights, 1.0):
            raise ValueError("priority weights must sum to one")
        thresholds = [
            self.info_threshold,
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        ]
        if thresholds != sorted(thresholds):
            raise ValueError("priority thresholds must be ordered")
        if self.close_score_threshold >= self.open_score_threshold:
            raise ValueError("close threshold must be below open threshold for hysteresis")
        return self
