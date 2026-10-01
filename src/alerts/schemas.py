"""Strict alert inputs and auditable decision-support outputs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AlertType(StrEnum):
    DELIVERY_RISK = "DELIVERY_RISK"
    ANOMALY = "ANOMALY"
    POWER_DERATING = "POWER_DERATING"
    ENERGY_CAPACITY_REDUCTION = "ENERGY_CAPACITY_REDUCTION"
    RACK_UNAVAILABLE = "RACK_UNAVAILABLE"
    PCS_UNAVAILABLE = "PCS_UNAVAILABLE"
    THERMAL_ISSUE = "THERMAL_ISSUE"
    POWER_TRACKING = "POWER_TRACKING"
    EFFICIENCY_DEGRADATION = "EFFICIENCY_DEGRADATION"
    COMMERCIAL_IMPACT = "COMMERCIAL_IMPACT"


class AlertStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class AlertInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp_utc: datetime
    asset_id: str = "BESS-001"
    component_id: str = "SITE"
    component_type: str = "SITE"
    available_discharge_power_mw: float = Field(ge=0)
    available_usable_energy_mwh: float = Field(ge=0)
    technical_availability: float = Field(ge=0, le=1)
    unavailable_racks: int = Field(default=0, ge=0)
    unavailable_pcs: int = Field(default=0, ge=0)
    limiting_factor: str | None = None
    failure_probability_6h: float | None = Field(default=None, ge=0, le=1)
    failure_probability_12h: float | None = Field(default=None, ge=0, le=1)
    failure_probability_24h: float | None = Field(default=None, ge=0, le=1)
    anomaly_score: float | None = Field(default=None, ge=0)
    detector_count: int = Field(default=0, ge=0)
    feature_completeness: float = Field(default=1.0, ge=0, le=1)
    revenue_at_risk_eur: float = 0.0
    revenue_at_risk_fraction: float | None = None
    market_mode: str | None = None
    price_source: str | None = None
    benchmark_disclaimer: str | None = None
    supporting_signals: tuple[str, ...] = ()
    source_versions: dict[str, str] = {}


class AlertRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alert_key: str
    asset_id: str
    component_id: str
    component_type: str
    alert_type: AlertType
    status: AlertStatus
    opened_at_utc: datetime
    updated_at_utc: datetime
    resolved_at_utc: datetime | None = None
    priority_score: float = Field(ge=0, le=1)
    priority_product: float = Field(ge=0, le=1)
    priority_weighted: float = Field(ge=0, le=1)
    priority_level: str
    failure_probability: float
    risk_horizon_hours: int | None
    model_confidence: float = Field(ge=0, le=1)
    technical_severity: float = Field(ge=0, le=1)
    commercial_severity: float = Field(ge=0, le=1)
    capacity_severity: float = Field(ge=0, le=1)
    anomaly_component: float = Field(ge=0, le=1)
    affected_power_mw: float = Field(ge=0)
    affected_energy_mwh: float = Field(ge=0)
    revenue_at_risk_eur: float
    anomaly_score: float | None = None
    limiting_factor: str | None = None
    supporting_signals: tuple[str, ...]
    source_versions: dict[str, str]
    market_mode: str | None = None
    price_source: str | None = None
    benchmark_disclaimer: str | None = None
    data_provenance: str = "DERIVED DECISION-SUPPORT OUTPUT"
