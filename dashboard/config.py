"""Dashboard configuration, independent of secrets."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    besspulse_api_url: str = "http://localhost:8000/api/v1"
    dashboard_api_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    dashboard_api_retries: int = Field(default=1, ge=0, le=3)
    dashboard_demo_fallback: bool = True
    dashboard_default_refresh_seconds: int = 30


@lru_cache
def get_dashboard_settings() -> DashboardSettings:
    return DashboardSettings()
