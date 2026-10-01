"""Typed delivery-risk target, split, model, and calibration configuration."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeliveryRiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizons_hours: tuple[int, ...] = (6, 12, 24)
    minimum_request_threshold_mw: float = Field(default=0.1, ge=0)
    failure_delivery_ratio_threshold: float = Field(default=0.95, gt=0, le=1)
    train_fraction: float = Field(default=0.55, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    calibration_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    minimum_training_rows: int = Field(default=120, ge=20)
    minimum_positive_rows: int = Field(default=12, ge=1)
    minimum_failure_events: int = Field(default=2, ge=1)
    minimum_calibration_rows: int = Field(default=30, ge=10)
    minimum_subgroup_rows: int = Field(default=20, ge=5)
    classification_threshold: float = Field(default=0.5, gt=0, lt=1)
    selection_pr_auc_tolerance: float = Field(default=0.01, ge=0)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    random_seed: int = 42
    logistic_c: float = Field(default=1.0, gt=0)
    rf_n_estimators: int = Field(default=100, ge=10)
    rf_max_depth: int = Field(default=8, ge=1)
    rf_min_samples_leaf: int = Field(default=4, ge=1)
    lgbm_n_estimators: int = Field(default=100, ge=10)
    lgbm_learning_rate: float = Field(default=0.05, gt=0)
    lgbm_num_leaves: int = Field(default=15, ge=2)
    xgb_n_estimators: int = Field(default=100, ge=10)
    xgb_learning_rate: float = Field(default=0.05, gt=0)
    xgb_max_depth: int = Field(default=4, ge=1)
    shap_sample_size: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def validate_configuration(self) -> "DeliveryRiskConfig":
        total = (
            self.train_fraction
            + self.validation_fraction
            + self.calibration_fraction
            + self.test_fraction
        )
        if not np.isclose(total, 1.0):
            raise ValueError("train/validation/calibration/test fractions must sum to one")
        if not self.horizons_hours or any(value <= 0 for value in self.horizons_hours):
            raise ValueError("horizons_hours must contain positive values")
        if len(set(self.horizons_hours)) != len(self.horizons_hours):
            raise ValueError("horizons_hours must be unique")
        return self
