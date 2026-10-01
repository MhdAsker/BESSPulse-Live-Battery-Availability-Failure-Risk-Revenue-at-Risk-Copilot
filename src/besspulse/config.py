"""Typed, non-secret simulation configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BatteryConfig(BaseModel):
    """Nameplate and operating-envelope parameters."""

    model_config = ConfigDict(extra="forbid")
    site_rated_power_mw: float = Field(default=20.0, gt=0)
    site_rated_energy_mwh: float = Field(default=40.0, gt=0)
    pcs_count: int = Field(default=4, gt=0)
    racks_per_pcs: int = Field(default=8, gt=0)
    telemetry_interval_minutes: int = Field(default=5, gt=0)
    soc_min: float = Field(default=0.10, ge=0, lt=1)
    soc_max: float = Field(default=0.90, gt=0, le=1)
    initial_soc: float = Field(default=0.50, ge=0, le=1)
    charge_efficiency: float = Field(default=0.96, gt=0, le=1)
    discharge_efficiency: float = Field(default=0.96, gt=0, le=1)
    nominal_rte: float = Field(default=0.92, gt=0, le=1)
    nominal_voltage_v: float = Field(default=1200.0, gt=0)

    @model_validator(mode="after")
    def validate_soc(self) -> "BatteryConfig":
        if not self.soc_min < self.soc_max:
            raise ValueError("soc_min must be below soc_max")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must be within SOC bounds")
        return self


class ThermalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambient_temperature_c: float = 20.0
    thermal_time_constant_hours: float = Field(default=2.0, gt=0)
    power_heating_c_per_hour: float = Field(default=4.0, ge=0)
    recent_power_weight: float = Field(default=0.25, ge=0, le=1)
    cooling_effectiveness: float = Field(default=1.0, gt=0)
    derating_start_c: float = 42.0
    shutdown_temperature_c: float = 55.0

    @model_validator(mode="after")
    def validate_temperatures(self) -> "ThermalConfig":
        if self.shutdown_temperature_c <= self.derating_start_c:
            raise ValueError("shutdown temperature must exceed derating start")
        return self


class DegradationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar_fade_per_day: float = Field(default=0.000002, ge=0)
    throughput_fade_per_efc: float = Field(default=0.00002, ge=0)
    hot_temperature_fade_per_hour: float = Field(default=0.000001, ge=0)
    depth_of_discharge_fade_per_unit_hour: float = Field(default=0.000001, ge=0)


class MarketConfig(BaseModel):
    """Non-secret ENTSO-E region and endpoint configuration."""

    model_config = ConfigDict(extra="forbid")
    market_region: str = "DE-LU"
    entsoe_domain: str = "10Y1001A1001A82H"
    market_timezone: str = "Europe/Berlin"
    entsoe_endpoint: str = "https://web-api.tp.entsoe.eu/api"


class FeatureConfig(BaseModel):
    """Central causal feature windows, thresholds, and peer policy."""

    model_config = ConfigDict(extra="forbid")
    power_residual_window: str = "1h"
    soc_volatility_window: str = "1h"
    soc_exposure_window: str = "6h"
    temperature_window: str = "1h"
    thermal_exposure_window: str = "6h"
    rte_trend_window: str = "6h"
    availability_window: str = "6h"
    alarm_window: str = "6h"
    market_price_window: str = "24h"
    market_volatility_window: str = "24h"
    market_alignment_tolerance: str = "60min"
    high_soc_threshold: float = Field(default=0.80, ge=0, le=1)
    low_soc_threshold: float = Field(default=0.20, ge=0, le=1)
    thermal_threshold_c: float = 35.0
    peer_min_population: int = Field(default=3, ge=1)
    peer_epsilon: float = Field(default=1e-9, gt=0)
    require_full_windows: bool = True

    @field_validator(
        "power_residual_window",
        "soc_volatility_window",
        "soc_exposure_window",
        "temperature_window",
        "thermal_exposure_window",
        "rte_trend_window",
        "availability_window",
        "alarm_window",
        "market_price_window",
        "market_volatility_window",
        "market_alignment_tolerance",
    )
    @classmethod
    def duration_is_positive(cls, value: str) -> str:
        import pandas as pd

        try:
            duration = pd.Timedelta(value)
        except ValueError as exc:
            raise ValueError(f"invalid feature duration: {value}") from exc
        if duration <= pd.Timedelta(0):
            raise ValueError("feature durations must be positive")
        return value

    @model_validator(mode="after")
    def soc_threshold_order(self) -> "FeatureConfig":
        if self.low_soc_threshold >= self.high_soc_threshold:
            raise ValueError("low SOC threshold must be below high SOC threshold")
        return self


class SimulationConfig(BaseModel):
    """Complete deterministic simulator configuration."""

    model_config = ConfigDict(extra="forbid")
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    simulation_seed: int = 42
    minimum_request_threshold_mw: float = Field(default=0.1, ge=0)
    delivery_failure_ratio: float = Field(default=0.95, gt=0, le=1)
    market: MarketConfig = Field(default_factory=MarketConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


def load_config(path: str | Path | None = None) -> SimulationConfig:
    """Load YAML configuration, or return validated defaults."""

    if path is None:
        return SimulationConfig()
    with Path(path).open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    return SimulationConfig.model_validate(raw or {})
