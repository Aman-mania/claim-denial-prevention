"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_auth_repository, get_current_user, settings
from api.schemas import LoginRequest, TokenResponse
from src.auth.repository import AuthRepository
from src.auth.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, repo: AuthRepository = Depends(get_auth_repository)) -> TokenResponse:
    user = repo.authenticate(email=payload.email, password=payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token(
        subject=user.email,
        role=user.role,
        secret=settings().jwt_secret,
        expires_minutes=settings().access_token_minutes,
        extra_claims={"uid": user.id},
    )
    repo.record_audit(user=user, action="auth.login", status="success")
    return TokenResponse(access_token=token, user=user.to_dict())


@router.get("/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"status": "success", "user": user}
