"""FastAPI dependencies for auth, repository, and agent loading."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import ApiSettings, get_settings
from src.auth.repository import AuthRepository
from src.auth.security import TokenError, verify_access_token

security = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def settings() -> ApiSettings:
    return get_settings()


@lru_cache(maxsize=1)
def get_auth_repository() -> AuthRepository:
    repo = AuthRepository(settings().database_url)
    repo.initialize()
    return repo


@lru_cache(maxsize=1)
def get_remediation_agent():
    from src.agent.remediation_agent import RemediationAgent

    s = settings()
    return RemediationAgent.load(
        gold_dir=s.gold_dir,
        models_dir=s.models_dir,
        vector_dir=s.vector_dir,
        enable_openai=s.enable_openai,
    )


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict[str, Any]:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        payload = verify_access_token(credentials.credentials, secret=settings().jwt_secret)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    repo = get_auth_repository()
    user = repo.get_user(str(payload.get("sub")))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return {**user.to_dict(), "token_role": payload.get("role")}


def require_role(*allowed_roles: str):
    allowed = {role.lower() for role in allowed_roles}

    def _dependency(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if str(user.get("role", "")).lower() not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _dependency
