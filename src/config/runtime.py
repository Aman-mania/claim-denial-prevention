"""Central runtime configuration for local, Docker, and AWS execution.

The project should not scatter `if local else aws` checks across business logic.
Instead, modules read this small runtime object and use the same service APIs.

Deployment mode is controlled by APP_ENV:
    local  -> normal laptop/venv execution
    docker -> local Docker Compose execution
    aws    -> EC2 Docker Compose + RDS/S3/Secrets-managed values
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeploymentEnv = Literal["local", "docker", "aws"]

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_VALID_ENVS = {"local", "docker", "aws"}


def is_truthy(value: str | bool | None, *, default: bool = False) -> bool:
    """Parse common truthy environment values safely."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in _TRUE_VALUES


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _app_env() -> DeploymentEnv:
    raw = (_env("APP_ENV", "local") or "local").strip().lower()
    if raw not in _VALID_ENVS:
        raise ValueError(f"APP_ENV must be one of {sorted(_VALID_ENVS)}, got {raw!r}")
    return raw  # type: ignore[return-value]


def _default_root() -> Path:
    # src/config/runtime.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _default_database_url(app_env: DeploymentEnv) -> str:
    # In local and Docker, SQLite is intentionally used for simple dev/testing.
    # In AWS, use RDS PostgreSQL by setting AUTH_DATABASE_URL in Secrets/env.
    if app_env == "aws":
        value = os.getenv("AUTH_DATABASE_URL", "").strip()
        if not value:
            raise RuntimeError(
                "APP_ENV=aws requires AUTH_DATABASE_URL to point to RDS PostgreSQL. "
                "Example: postgresql://user:password@host:5432/claim_denial"
            )
        return value
    return os.getenv("AUTH_DATABASE_URL", "sqlite:///data/auth/auth.db").strip()


def _default_api_base_url(app_env: DeploymentEnv) -> str:
    # Docker/AWS compose networks can use the service name `api`.
    if app_env in {"docker", "aws"}:
        return "http://api:8000"
    return "http://localhost:8000"


@dataclass(frozen=True)
class RuntimeSettings:
    app_env: DeploymentEnv
    root_dir: Path
    gold_dir: Path
    models_dir: Path
    vector_dir: Path
    policies_dir: Path
    logs_dir: Path
    auth_database_url: str
    jwt_secret: str
    access_token_minutes: int
    enable_openai: bool
    openai_model: str
    openai_api_key_configured: bool
    ui_api_base_url: str
    ui_api_timeout_seconds: float
    cors_origins: list[str]
    aws_region: str
    s3_bucket_name: str | None
    use_s3_artifacts: bool

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    @property
    def is_docker(self) -> bool:
        return self.app_env == "docker"

    @property
    def is_aws(self) -> bool:
        return self.app_env == "aws"

    @property
    def auth_backend(self) -> str:
        url = self.auth_database_url
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return "postgresql"
        if url.startswith("sqlite"):
            return "sqlite"
        return "custom"


def get_runtime_settings() -> RuntimeSettings:
    app_env = _app_env()
    root = Path(_env("CLAIM_DENIAL_ROOT", str(_default_root())) or str(_default_root())).resolve()

    origins = [
        item.strip()
        for item in (_env("API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501") or "").split(",")
        if item.strip()
    ]

    return RuntimeSettings(
        app_env=app_env,
        root_dir=root,
        gold_dir=_resolve_under_root(root, _env("GOLD_DIR", "data/gold") or "data/gold"),
        models_dir=_resolve_under_root(root, _env("MODELS_DIR", "models") or "models"),
        vector_dir=_resolve_under_root(root, _env("RAG_VECTOR_DIR", "data/vector_store") or "data/vector_store"),
        policies_dir=_resolve_under_root(root, _env("POLICIES_DIR", "data/policies") or "data/policies"),
        logs_dir=_resolve_under_root(root, _env("LOGS_DIR", "logs") or "logs"),
        auth_database_url=_default_database_url(app_env),
        jwt_secret=_env("JWT_SECRET", "local-dev-change-me-before-deployment") or "local-dev-change-me-before-deployment",
        access_token_minutes=int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "120") or "120"),
        enable_openai=is_truthy(_env("ENABLE_OPENAI_AGENT_OUTPUT"), default=False),
        openai_model=_env("OPENAI_AGENT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_api_key_configured=bool(_env("OPENAI_API_KEY")),
        ui_api_base_url=(_env("CLAIM_DENIAL_API_BASE_URL") or _default_api_base_url(app_env)).rstrip("/"),
        ui_api_timeout_seconds=float(_env("CLAIM_DENIAL_UI_API_TIMEOUT_SECONDS", "45") or "45"),
        cors_origins=origins,
        aws_region=_env("AWS_REGION", "us-east-1") or "us-east-1",
        s3_bucket_name=_env("S3_BUCKET_NAME"),
        use_s3_artifacts=is_truthy(_env("USE_S3_ARTIFACTS"), default=False),
    )
