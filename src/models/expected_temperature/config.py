"""Expected-temperature model configuration."""

from pydantic import ConfigDict

from models.common import ExpectedModelConfig


class ExpectedTemperatureConfig(ExpectedModelConfig):
    model_config = ConfigDict(extra="forbid")
    model_version: str = "expected_temperature_v1"
    temperature_lag_short: str = "5min"
    temperature_lag_long: str = "15min"
    recent_window: str = "30min"
    residual_persistence_threshold: float = 1.0
