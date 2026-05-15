"""Small dependency-light auth/security helpers.

This module intentionally avoids PyJWT/passlib so the local project can run on
restricted corporate machines. Tokens use standard JWT HS256 shape
(header.payload.signature) and are suitable for the project demo. On AWS, the
same code can use a JWT secret from Secrets Manager.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

_DEFAULT_ITERATIONS = 260_000
_DEFAULT_ALGORITHM = "HS256"


class TokenError(ValueError):
    """Raised when an access token is invalid or expired."""


@dataclass(frozen=True)
class PasswordHashConfig:
    iterations: int = _DEFAULT_ITERATIONS
    algorithm: str = "pbkdf2_sha256"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def hash_password(password: str, *, config: PasswordHashConfig | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256.

    Format: pbkdf2_sha256$iterations$salt$hash
    """
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    cfg = config or PasswordHashConfig()
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, cfg.iterations)
    return f"{cfg.algorithm}${cfg.iterations}${_b64url_encode(salt)}${_b64url_encode(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain password against a stored PBKDF2 hash."""
    try:
        algorithm, iterations_text, salt_text, digest_text = str(stored_hash).split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = _b64url_decode(salt_text)
        expected = _b64url_decode(digest_text)
        actual = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def get_jwt_secret(secret: str | None = None) -> str:
    """Return the configured JWT secret or a local-development fallback.

    For AWS deployment, set JWT_SECRET through Secrets Manager/.env. The fallback
    is intentionally deterministic only for local development; do not use it in
    production/demo hosting.
    """
    value = secret or os.getenv("JWT_SECRET") or os.getenv("CLAIM_DENIAL_JWT_SECRET")
    if value:
        return value
    return "local-dev-change-me-before-deployment"


def create_access_token(
    *,
    subject: str,
    role: str,
    secret: str | None = None,
    expires_minutes: int = 120,
    issuer: str = "claim-denial-api",
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a compact HS256 JWT access token."""
    now = int(time.time())
    header = {"alg": _DEFAULT_ALGORITHM, "typ": "JWT"}
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": str(role),
        "iss": issuer,
        "iat": now,
        "exp": now + int(expires_minutes) * 60,
    }
    if extra_claims:
        payload.update(extra_claims)
    signing_input = ".".join([
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url_encode(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")),
    ])
    signature = hmac.new(get_jwt_secret(secret).encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def verify_access_token(
    token: str,
    *,
    secret: str | None = None,
    issuer: str = "claim-denial-api",
    leeway_seconds: int = 0,
) -> dict[str, Any]:
    """Verify an HS256 JWT and return its payload."""
    try:
        header_b64, payload_b64, signature_b64 = str(token).split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}"
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != _DEFAULT_ALGORITHM:
            raise TokenError("unsupported token algorithm")
        expected_sig = hmac.new(get_jwt_secret(secret).encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        actual_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise TokenError("invalid token signature")
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("iss") != issuer:
            raise TokenError("invalid token issuer")
        exp = int(payload.get("exp", 0))
        if exp + int(leeway_seconds) < int(time.time()):
            raise TokenError("token expired")
        if not payload.get("sub") or not payload.get("role"):
            raise TokenError("token missing subject or role")
        return payload
    except TokenError:
        raise
    except Exception as exc:
        raise TokenError("invalid access token") from exc
