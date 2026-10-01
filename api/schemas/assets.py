"""Asset listing and aggregated status contracts."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, Freshness, Provenance


class AssetSummary(APIModel):
    asset_id: str
    name: str
    rated_power_mw: float = Field(gt=0)
    rated_energy_mwh: float = Field(gt=0)
    pcs_count: int = Field(gt=0)
    rack_count: int = Field(gt=0)
    status: str
    data_provenance: Provenance


class AssetStatus(APIModel):
    asset_id: str
    timestamp_utc: datetime
    rated_power_mw: float = Field(gt=0)
    rated_energy_mwh: float = Field(gt=0)
    available_power_mw: float | None = Field(default=None, ge=0)
    available_energy_mwh: float | None = Field(default=None, ge=0)
    site_soc: float | None = Field(default=None, ge=0, le=1)
    rte: float | None = Field(default=None, ge=0, le=1)
    technical_availability: float | None = Field(default=None, ge=0, le=1)
    requested_power_availability: float | None = Field(default=None, ge=0, le=1)
    available_racks: int | None = Field(default=None, ge=0)
    available_pcs: int | None = Field(default=None, ge=0)
    active_alert_count: int = Field(ge=0)
    highest_priority_alert: str | None
    failure_probability_6h: float | None = Field(default=None, ge=0, le=1)
    failure_probability_12h: float | None = Field(default=None, ge=0, le=1)
    failure_probability_24h: float | None = Field(default=None, ge=0, le=1)
    revenue_at_risk_eur: float | None
    telemetry_timestamp_utc: datetime | None
    risk_prediction_timestamp_utc: datetime | None
    availability_timestamp_utc: datetime | None
    market_timestamp_utc: datetime | None
    commercial_timestamp_utc: datetime | None
    freshness: dict[str, Freshness]
    provenance: tuple[Provenance, ...]
