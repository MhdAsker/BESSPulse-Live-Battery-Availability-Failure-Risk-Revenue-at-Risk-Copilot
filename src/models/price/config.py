"""Typed configuration for compact DE-LU day-ahead price forecasting."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PriceModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    forecast_horizon_hours: int = Field(default=24, ge=1)
    target_name: str = "day_ahead_price_eur_per_mwh"
    market_region: str = "DE-LU"
    market_timezone: str = "Europe/Berlin"
    lag_durations: tuple[str, ...] = ("24h", "168h")
    rolling_windows: tuple[str, ...] = ("24h", "168h")
    minimum_training_history_hours: int = Field(default=24 * 28, ge=168)
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)
    test_fraction: float = Field(default=0.20, gt=0, lt=1)
    selection_tolerance_pct: float = Field(default=2.0, ge=0)
    high_price_training_quantile: float = Field(default=0.95, gt=0.5, lt=1)
    random_seed: int = 42
    lgbm_n_estimators: int = Field(default=160, ge=10)
    lgbm_learning_rate: float = Field(default=0.04, gt=0)
    lgbm_num_leaves: int = Field(default=15, ge=2)
    lgbm_max_depth: int = Field(default=-1, ge=-1)
    lgbm_min_child_samples: int = Field(default=20, ge=1)
    lgbm_subsample: float = Field(default=0.9, gt=0, le=1)
    lgbm_colsample_bytree: float = Field(default=0.9, gt=0, le=1)
    lgbm_reg_alpha: float = Field(default=0.0, ge=0)
    lgbm_reg_lambda: float = Field(default=0.2, ge=0)
    quantiles_enabled: bool = True
    quantile_levels: tuple[float, ...] = (0.10, 0.50, 0.90)
    walk_forward_step_hours: int = Field(default=24, ge=1)
    walk_forward_max_origins: int = Field(default=12, ge=1)
    feature_set_version: str = "price_features_v1"
    model_version: str = "price_forecast_v1"

    @field_validator("lag_durations", "rolling_windows")
    @classmethod
    def durations_are_positive(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(pd.Timedelta(value) <= pd.Timedelta(0) for value in values):
            raise ValueError("lag and rolling durations must be non-empty and positive")
        return values

    @model_validator(mode="after")
    def validate_config(self) -> "PriceModelConfig":
        if not np.isclose(self.train_fraction + self.validation_fraction + self.test_fraction, 1.0):
            raise ValueError("train/validation/test fractions must sum to one")
        if tuple(sorted(self.quantile_levels)) != self.quantile_levels:
            raise ValueError("quantile levels must be sorted")
        if any(not 0 < value < 1 for value in self.quantile_levels):
            raise ValueError("quantiles must be strictly between zero and one")
        return self
