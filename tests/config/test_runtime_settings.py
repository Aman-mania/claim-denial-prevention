from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _reload(monkeypatch, **env):
    keys = [
        "APP_ENV", "CLAIM_DENIAL_ROOT", "AUTH_DATABASE_URL", "CLAIM_DENIAL_API_BASE_URL",
        "ENABLE_OPENAI_AGENT_OUTPUT", "OPENAI_API_KEY", "S3_BUCKET_NAME", "USE_S3_ARTIFACTS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import src.config.runtime as runtime
    importlib.reload(runtime)
    return runtime


def test_local_runtime_defaults_to_sqlite_and_localhost(monkeypatch):
    runtime = _reload(monkeypatch, APP_ENV="local")
    settings = runtime.get_runtime_settings()
    assert settings.app_env == "local"
    assert settings.auth_backend == "sqlite"
    assert settings.ui_api_base_url == "http://localhost:8000"


def test_docker_runtime_defaults_to_api_service_name(monkeypatch):
    runtime = _reload(monkeypatch, APP_ENV="docker")
    settings = runtime.get_runtime_settings()
    assert settings.app_env == "docker"
    assert settings.ui_api_base_url == "http://api:8000"


def test_aws_runtime_requires_auth_database_url(monkeypatch):
    runtime = _reload(monkeypatch, APP_ENV="aws")
    with pytest.raises(RuntimeError, match="AUTH_DATABASE_URL"):
        runtime.get_runtime_settings()


def test_aws_runtime_uses_postgres_url(monkeypatch):
    runtime = _reload(
        monkeypatch,
        APP_ENV="aws",
        AUTH_DATABASE_URL="postgresql://user:pass@example.com:5432/claim_denial",
        OPENAI_API_KEY="sk-test",
        ENABLE_OPENAI_AGENT_OUTPUT="true",
    )
    settings = runtime.get_runtime_settings()
    assert settings.is_aws
    assert settings.auth_backend == "postgresql"
    assert settings.enable_openai is True
    assert settings.openai_api_key_configured is True
