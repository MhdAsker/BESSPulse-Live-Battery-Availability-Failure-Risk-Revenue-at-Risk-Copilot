"""Alert filtering and pagination response contracts."""

from datetime import datetime

from pydantic import Field

from alerts.schemas import AlertStatus, AlertType
from api.schemas.common import APIModel, PageMetadata, Provenance


class AlertResponse(APIModel):
    alert_id: str
    asset_id: str
    component: str
    component_type: str
    alert_type: AlertType
    status: AlertStatus
    priority_score: float = Field(ge=0, le=1)
    priority_level: str
    failure_probability: float = Field(ge=0, le=1)
    risk_horizon: int | None
    model_confidence: float = Field(ge=0, le=1)
    technical_severity: float = Field(ge=0, le=1)
    commercial_severity: float = Field(ge=0, le=1)
    affected_power_mw: float = Field(ge=0)
    affected_energy_mwh: float = Field(ge=0)
    revenue_at_risk_eur: float
    anomaly_score: float | None = Field(default=None, ge=0)
    anomaly_score_semantics: str = "detector score; not a probability"
    limiting_factor: str | None
    supporting_signals: tuple[str, ...]
    opened_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    source_versions: dict[str, str]
    data_provenance: Provenance


class AlertPage(PageMetadata):
    items: tuple[AlertResponse, ...]
