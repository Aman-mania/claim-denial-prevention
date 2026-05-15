"""Pydantic schemas for FastAPI request/response bodies."""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic comes with FastAPI
    BaseModel = object  # type: ignore
    def Field(default=None, **kwargs):  # type: ignore
        return default


class LoginRequest(BaseModel):
    email: str = Field(..., examples=["analyst@example.com"])
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class ClaimRequest(BaseModel):
    claim_id: str
    patient_id: str
    provider_id: str
    diagnosis_code: str | None = None
    procedure_code: str | None = None
    billed_amount: float | None = None
    specialty: str | None = None
    location: str | None = None


class ApiEnvelope(BaseModel):
    status: str
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
