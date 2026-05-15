#!/usr/bin/env python3
"""Preflight checks for Phase 5 FastAPI/auth setup."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    print("Week 7 Phase 5 API preflight")
    print(f"  fastapi: {'OK' if _available('fastapi') else 'MISSING'}")
    print(f"  uvicorn: {'OK' if _available('uvicorn') else 'MISSING'}")
    print(f"  psycopg: {'OK' if _available('psycopg') else 'MISSING / only needed for RDS PostgreSQL'}")
    print(f"  openai:  {'OK' if _available('openai') else 'MISSING / optional presentation layer'}")
    print(f"  AUTH_DATABASE_URL: {os.getenv('AUTH_DATABASE_URL', 'sqlite:///data/auth/auth.db')}")
    print(f"  JWT_SECRET set: {'yes' if os.getenv('JWT_SECRET') else 'no - local fallback will be used'}")

    required_local = [
        Path("data/gold/inference_artifacts.json"),
        Path("models/training_report.json"),
        Path("data/vector_store/policy_metadata.json"),
    ]
    print("  Artifact status:")
    for path in required_local:
        print(f"    {'OK' if path.exists() else 'MISSING'} {path}")

    if not _available('fastapi') or not _available('uvicorn'):
        print("\nInstall API deps when your network allows:")
        print("  python -m pip install -r requirements-api.txt")
        return 2
    print("\nAPI preflight passed for local FastAPI startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
