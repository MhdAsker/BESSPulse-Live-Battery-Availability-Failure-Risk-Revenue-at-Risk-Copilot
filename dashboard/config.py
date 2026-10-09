"""Dashboard configuration, independent of secrets."""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    besspulse_api_url: str = "http://localhost:8000/api/v1"
    dashboard_api_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    dashboard_api_retries: int = Field(default=1, ge=0, le=3)
    dashboard_demo_fallback: bool = True
    dashboard_default_refresh_seconds: int = 30

    @model_validator(mode="after")
    def validate_production_api(self) -> "DashboardSettings":
        parsed = urlparse(self.besspulse_api_url)
        if self.app_env == "production":
            if parsed.hostname in {"localhost", "127.0.0.1"}:
                raise ValueError("Production dashboard cannot use a localhost API URL.")
            if parsed.scheme != "https" and parsed.hostname != "api":
                raise ValueError("Production dashboard requires an HTTPS API URL.")
            if self.dashboard_demo_fallback:
                raise ValueError("Production dashboard demo fallback must be disabled.")
        return self


@lru_cache
def get_dashboard_settings() -> DashboardSettings:
    return DashboardSettings()
