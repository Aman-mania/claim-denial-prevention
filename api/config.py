"""API configuration loaded from centralized runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.runtime import get_runtime_settings


@dataclass(frozen=True)
class ApiSettings:
    app_env: str
    root_dir: Path
    gold_dir: Path
    models_dir: Path
    vector_dir: Path
    database_url: str
    auth_backend: str
    jwt_secret: str
    access_token_minutes: int
    enable_openai: bool
    openai_model: str
    openai_api_key_configured: bool
    cors_origins: list[str]


def get_settings() -> ApiSettings:
    runtime = get_runtime_settings()
    return ApiSettings(
        app_env=runtime.app_env,
        root_dir=runtime.root_dir,
        gold_dir=runtime.gold_dir,
        models_dir=runtime.models_dir,
        vector_dir=runtime.vector_dir,
        database_url=runtime.auth_database_url,
        auth_backend=runtime.auth_backend,
        jwt_secret=runtime.jwt_secret,
        access_token_minutes=runtime.access_token_minutes,
        enable_openai=runtime.enable_openai,
        openai_model=runtime.openai_model,
        openai_api_key_configured=runtime.openai_api_key_configured,
        cors_origins=runtime.cors_origins,
    )
