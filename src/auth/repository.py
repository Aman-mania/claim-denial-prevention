"""Authentication repository for local SQLite and AWS RDS PostgreSQL.

Local tests/development can use SQLite with no extra dependency. AWS deployment
uses an RDS PostgreSQL URL and imports psycopg lazily only when needed.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.auth.security import hash_password, verify_password

VALID_ROLES = {"developer", "analyst"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_database_url(url: str | None) -> str:
    if url and url.strip():
        return url.strip()
    return os.getenv("AUTH_DATABASE_URL", "sqlite:///data/auth/auth.db").strip()


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    role: str
    is_active: bool
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AuthRepository:
    """Small DB access layer for users, roles, and audit events."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = _normalize_database_url(database_url)

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql://") or self.database_url.startswith("postgres://")

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except Exception as exc:  # pragma: no cover - requires optional dependency
                raise RuntimeError(
                    "PostgreSQL auth requires psycopg. Install API deps with: python -m pip install -r requirements-api.txt"
                ) from exc
            conn = psycopg.connect(self.database_url, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
        else:
            path = self._sqlite_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _sqlite_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1))
        if self.database_url.startswith("sqlite://"):
            return Path(self.database_url.replace("sqlite://", "", 1))
        return Path(self.database_url)

    def _placeholder(self) -> str:
        return "%s" if self.is_postgres else "?"

    def initialize(self) -> None:
        """Create auth and audit tables if they do not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    email TEXT,
                    action TEXT NOT NULL,
                    claim_id TEXT,
                    status TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_user(self, *, email: str, password: str, role: str, overwrite: bool = False) -> AuthUser:
        email_norm = str(email).strip().lower()
        role_norm = str(role).strip().lower()
        if role_norm not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        if not email_norm:
            raise ValueError("email is required")
        if not password:
            raise ValueError("password is required")

        existing = self.get_user_with_hash(email_norm)
        now = _utc_now()
        if existing and not overwrite:
            return self._public_user(existing)

        password_hash = hash_password(password)
        with self._connect() as conn:
            if existing:
                ph = self._placeholder()
                conn.execute(
                    f"UPDATE users SET password_hash={ph}, role={ph}, is_active={ph}, updated_at={ph} WHERE email={ph}",
                    (password_hash, role_norm, True, now, email_norm),
                )
            else:
                user_id = str(uuid.uuid4())
                ph = self._placeholder()
                conn.execute(
                    f"INSERT INTO users (id, email, password_hash, role, is_active, created_at, updated_at) VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                    (user_id, email_norm, password_hash, role_norm, True, now, now),
                )
        user = self.get_user(email_norm)
        if user is None:  # pragma: no cover - defensive
            raise RuntimeError("user creation failed")
        return user

    def get_user_with_hash(self, email: str) -> dict[str, Any] | None:
        email_norm = str(email).strip().lower()
        ph = self._placeholder()
        with self._connect() as conn:
            cur = conn.execute(f"SELECT * FROM users WHERE email={ph}", (email_norm,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)

    def _public_user(self, row: dict[str, Any]) -> AuthUser:
        return AuthUser(
            id=str(row["id"]),
            email=str(row["email"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
            updated_at=str(row.get("updated_at")) if row.get("updated_at") is not None else None,
        )

    def get_user(self, email: str) -> AuthUser | None:
        row = self.get_user_with_hash(email)
        return self._public_user(row) if row else None

    def authenticate(self, *, email: str, password: str) -> AuthUser | None:
        row = self.get_user_with_hash(email)
        if not row or not bool(row.get("is_active")):
            return None
        if not verify_password(password, str(row.get("password_hash"))):
            return None
        return self._public_user(row)

    def record_audit(
        self,
        *,
        user: AuthUser | dict[str, Any] | None,
        action: str,
        claim_id: str | None = None,
        status: str | None = None,
        metadata: str | None = None,
    ) -> None:
        user_id = None
        email = None
        if isinstance(user, AuthUser):
            user_id = user.id
            email = user.email
        elif isinstance(user, dict):
            user_id = user.get("id") or user.get("sub")
            email = user.get("email")
        ph = self._placeholder()
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO audit_events (id, user_id, email, action, claim_id, status, metadata, created_at) VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (str(uuid.uuid4()), user_id, email, action, claim_id, status, metadata, _utc_now()),
            )
