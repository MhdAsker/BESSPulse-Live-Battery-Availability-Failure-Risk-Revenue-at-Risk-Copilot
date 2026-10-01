"""Typed anomaly detector, persistence, event, and evaluation configuration."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_fraction: float = Field(default=0.50, gt=0, lt=1)
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    test_fraction: float = Field(default=0.25, gt=0, lt=1)
    random_seed: int = 42
    minimum_training_rows: int = Field(default=500, ge=20)
    minimum_peer_count: int = Field(default=3, ge=1)
    mad_epsilon: float = Field(default=1e-9, gt=0)
    robust_z_candidates: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)
    residual_z_candidates: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)
    isolation_quantile_candidates: tuple[float, ...] = (0.95, 0.975, 0.99)
    persistence_candidates: tuple[int, ...] = (1, 2, 3, 6)
    max_false_alerts_per_day: float = Field(default=4.0, ge=0)
    merge_gap: str = "10min"
    early_warning_window: str = "2h"
    post_fault_grace: str = "30min"
    iforest_n_estimators: int = Field(default=100, ge=10)
    iforest_max_samples: int = Field(default=256, ge=16)
    iforest_contamination: float = Field(default=0.02, gt=0, lt=0.5)
    engineering_temperature_c: float = 42.0
    engineering_temperature_vs_ambient_c: float = Field(default=15.0, gt=0)
    engineering_temperature_spread_c: float = Field(default=5.0, gt=0)
    engineering_temperature_ramp_c: float = Field(default=2.0, gt=0)
    engineering_voltage_spread_v: float = Field(default=15.0, gt=0)
    engineering_min_rte: float = Field(default=0.82, gt=0, le=1)
    engineering_power_tracking_error_mw: float = Field(default=0.05, gt=0)
    ensemble_vote_threshold: int = Field(default=2, ge=1, le=4)

    @model_validator(mode="after")
    def validate_config(self) -> "AnomalyConfig":
        if not np.isclose(self.train_fraction + self.validation_fraction + self.test_fraction, 1):
            raise ValueError("train/validation/test fractions must sum to one")
        for value in [self.merge_gap, self.early_warning_window, self.post_fault_grace]:
            if pd.Timedelta(value) < pd.Timedelta(0):
                raise ValueError("anomaly time windows cannot be negative")
        if any(value < 1 for value in self.persistence_candidates):
            raise ValueError("persistence candidates must be positive")
        return self
