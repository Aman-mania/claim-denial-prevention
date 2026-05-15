"""API configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    root_dir: Path
    gold_dir: Path
    models_dir: Path
    vector_dir: Path
    database_url: str
    jwt_secret: str
    access_token_minutes: int
    enable_openai: bool
    openai_model: str
    cors_origins: list[str]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> ApiSettings:
    root = Path(os.getenv("CLAIM_DENIAL_ROOT", Path(__file__).resolve().parents[1])).resolve()
    origins = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
    return ApiSettings(
        root_dir=root,
        gold_dir=(root / os.getenv("GOLD_DIR", "data/gold")).resolve(),
        models_dir=(root / os.getenv("MODELS_DIR", "models")).resolve(),
        vector_dir=(root / os.getenv("RAG_VECTOR_DIR", "data/vector_store")).resolve(),
        database_url=os.getenv("AUTH_DATABASE_URL", "sqlite:///data/auth/auth.db"),
        jwt_secret=os.getenv("JWT_SECRET", "local-dev-change-me-before-deployment"),
        access_token_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120")),
        enable_openai=_bool_env("ENABLE_OPENAI_AGENT_OUTPUT", False),
        openai_model=os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini"),
        cors_origins=origins,
    )
