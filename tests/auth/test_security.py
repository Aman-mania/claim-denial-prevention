from __future__ import annotations

import pytest

from src.auth.security import TokenError, create_access_token, hash_password, verify_access_token, verify_password


def test_password_hash_round_trip():
    stored = hash_password("secret-password")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("secret-password", stored)
    assert not verify_password("wrong", stored)


def test_access_token_round_trip():
    token = create_access_token(subject="analyst@example.com", role="analyst", secret="test-secret", expires_minutes=5)
    payload = verify_access_token(token, secret="test-secret")
    assert payload["sub"] == "analyst@example.com"
    assert payload["role"] == "analyst"


def test_access_token_rejects_wrong_secret():
    token = create_access_token(subject="dev@example.com", role="developer", secret="good-secret")
    with pytest.raises(TokenError):
        verify_access_token(token, secret="bad-secret")
