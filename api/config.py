"""Typed, non-secret API configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: Literal["development", "test", "production"] = "development"
    build_commit: str = "unknown"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_log_level: str = Field(
        default="INFO", validation_alias=AliasChoices("API_LOG_LEVEL", "LOG_LEVEL")
    )
    api_cors_origins: tuple[str, ...] = (
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )
    enable_simulation_control_api: bool = False
    api_max_page_size: int = Field(default=200, ge=1, le=1000)
    api_default_page_size: int = Field(default=50, ge=1, le=200)
    api_stale_after_seconds: int = Field(default=900, gt=0)
    api_max_query_days: int = Field(default=31, ge=1, le=366)
    database_url: str | None = None
    rag_enabled: bool = False
    rag_backend: Literal["local"] = "local"
    database_startup_attempts: int = Field(default=3, ge=1, le=10)
    database_retry_delay_seconds: float = Field(default=1.0, ge=0, le=10)
    api_json_logs: bool = False

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> "APISettings":
        if self.app_env == "production":
            if not self.database_url or self.database_url.startswith("sqlite"):
                raise ValueError("Production requires a PostgreSQL DATABASE_URL.")
            if "*" in self.api_cors_origins:
                raise ValueError("Production CORS cannot allow wildcard origins.")
            if self.enable_simulation_control_api:
                raise ValueError("Simulation controls must be disabled in production.")
        return self

    @property
    def effective_database_url(self) -> str:
        return self.database_url or "sqlite:///data/besspulse.db"


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
