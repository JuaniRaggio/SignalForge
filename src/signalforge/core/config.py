"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "SignalForge"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Database
    database_url: str = (
        "postgresql+asyncpg://signalforge:signalforge_dev@localhost:5434/signalforge"
    )

    # Redis
    redis_url: str = "redis://localhost:6380/0"

    # JWT
    jwt_secret_key: str = "your-super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Celery
    celery_broker_url: str = "redis://localhost:6380/1"
    celery_result_backend: str = "redis://localhost:6380/2"

    # API
    api_v1_prefix: str = "/api/v1"

    # CORS
    cors_allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Rate Limiting
    rate_limit_per_minute: int = 5
    rate_limit_window_seconds: int = 60

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, v: str, info) -> str:
        """Validate JWT secret key is not using default value in production."""
        app_env = info.data.get("app_env", "development")
        if app_env == "production" and v == "your-super-secret-key-change-in-production":
            raise ValueError(
                "JWT_SECRET_KEY must be set to a secure value in production. "
                "Set the JWT_SECRET_KEY environment variable to a secure random string."
            )
        if app_env == "production" and len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters long in production."
            )
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
