from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "backend" / "sales_forecasting.db"
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

TRIVIAL_SECRETS = {
    "your-secret-key-here",
    "change-me",
    "secret",
    "password",
    "development-secret-key",
    "dev-fallback-secret-key-change-in-production",
    "dev-insecure-secret-key-for-local-testing-only",
    "123456",
    "admin",
}


class Settings(BaseSettings):
    environment: Literal["development", "test", "production"] = "development"
    secret_key: str = "dev-insecure-secret-key-for-local-testing-only"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    database_url: str = DEFAULT_SQLITE_URL
    allowed_origins: str | list[str] = "http://localhost:5173,http://127.0.0.1:5173"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    admin_username: str = "admin"
    admin_email: str = "admin@smart-sales.local"
    admin_password: str | None = Field(default=None, repr=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def cors_allowed_origins(self) -> list[str]:
        if isinstance(self.allowed_origins, list):
            return [origin.strip() for origin in self.allowed_origins if origin.strip()]
        if isinstance(self.allowed_origins, str):
            raw = self.allowed_origins.strip()
            if raw.startswith("[") and raw.endswith("]"):
                import json
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(o).strip() for o in parsed if str(o).strip()]
                except Exception:
                    pass
            return [part.strip() for part in raw.split(",") if part.strip()]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    @model_validator(mode="after")
    def validate_production_configuration(self) -> Settings:
        env = self.environment.lower()
        if env == "production":
            # 1. Validate secret key
            key = (self.secret_key or "").strip()
            if not key or key.lower() in TRIVIAL_SECRETS or len(key) < 16:
                raise ValueError(
                    "Production configuration error: SECRET_KEY must be set to a secure, non-trivial secret (minimum 16 characters)."
                )

            # 2. Validate database URL
            db_url = (self.database_url or "").strip()
            if not db_url or db_url.startswith("sqlite"):
                raise ValueError(
                    "Production configuration error: DATABASE_URL must be specified and must not use local SQLite file for production deployment."
                )

            # 3. Validate CORS origins: no wildcard "*" with credentials
            origins = self.cors_allowed_origins
            if "*" in origins:
                raise ValueError(
                    "Production configuration error: Wildcard '*' is not permitted in ALLOWED_ORIGINS when credentialed authentication is enabled."
                )

        return self


settings = Settings()
