from __future__ import annotations

from src.auth.repository import AuthRepository


def test_auth_repository_sqlite_creates_and_authenticates_user(tmp_path):
    repo = AuthRepository(f"sqlite:///{tmp_path / 'auth.db'}")
    repo.initialize()
    created = repo.create_user(email="Analyst@Example.com", password="pw123", role="analyst")
    assert created.email == "analyst@example.com"
    assert created.role == "analyst"

    assert repo.authenticate(email="analyst@example.com", password="pw123") is not None
    assert repo.authenticate(email="analyst@example.com", password="wrong") is None


def test_auth_repository_records_audit_event(tmp_path):
    repo = AuthRepository(f"sqlite:///{tmp_path / 'auth.db'}")
    repo.initialize()
    user = repo.create_user(email="dev@example.com", password="pw123", role="developer")
    repo.record_audit(user=user, action="claims.recommend", claim_id="C001", status="success")
