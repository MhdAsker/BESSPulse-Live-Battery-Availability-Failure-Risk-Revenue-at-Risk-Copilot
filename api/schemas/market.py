"""Latest observed and forecast market context."""

from datetime import datetime

from api.schemas.common import APIModel, Freshness, Provenance


class ObservedMarketFields(APIModel):
    timestamp_utc: datetime
    day_ahead_price_eur_per_mwh: float | None = None
    load_mw: float | None = None
    wind_generation_mw: float | None = None
    solar_generation_mw: float | None = None
    renewable_generation_mw: float | None = None


class ForecastMarketFields(APIModel):
    forecast_origin_utc: datetime | None = None
    target_timestamp_utc: datetime | None = None
    day_ahead_price_eur_per_mwh: float | None = None
    load_forecast_mw: float | None = None
    wind_forecast_mw: float | None = None
    solar_forecast_mw: float | None = None


class DerivedMarketFields(APIModel):
    residual_load_mw: float | None = None
    renewable_share: float | None = None


class MarketLatestResponse(APIModel):
    market_region: str
    observed: ObservedMarketFields | None
    forecast: ForecastMarketFields | None
    derived: DerivedMarketFields
    freshness: Freshness
    observed_provenance: Provenance | None
    forecast_provenance: Provenance | None
    derived_provenance: Provenance
