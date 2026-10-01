"""Anomaly history contracts."""

from datetime import datetime

from pydantic import Field

from api.schemas.common import APIModel, PageMetadata, Provenance


class AnomalyEventResponse(APIModel):
    event_id: str
    rack_id: str
    detector: str
    detector_version: str
    start: datetime
    end: datetime
    peak_score: float = Field(ge=0)
    mean_score: float = Field(ge=0)
    supporting_signals: tuple[str, ...]
    status: str
    provenance: Provenance


class AnomalyPage(PageMetadata):
    items: tuple[AnomalyEventResponse, ...]
