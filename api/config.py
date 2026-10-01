"""Typed, non-secret API configuration."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_log_level: str = "INFO"
    api_cors_origins: tuple[str, ...] = (
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )
    enable_simulation_control_api: bool = False
    api_max_page_size: int = Field(default=200, ge=1, le=1000)
    api_default_page_size: int = Field(default=50, ge=1, le=200)
    api_stale_after_seconds: int = Field(default=900, gt=0)
    api_max_query_days: int = Field(default=31, ge=1, le=366)
    database_url: str = "sqlite:///data/besspulse.db"

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
