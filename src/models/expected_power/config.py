"""Expected-power model configuration."""

from pydantic import ConfigDict

from models.common import ExpectedModelConfig


class ExpectedPowerConfig(ExpectedModelConfig):
    model_config = ConfigDict(extra="forbid")
    model_version: str = "expected_power_v1"
    active_intervals_only_for_training: bool = True
    minimum_request_threshold_mw: float = 0.1
    rated_power_mw: float = 20.0
