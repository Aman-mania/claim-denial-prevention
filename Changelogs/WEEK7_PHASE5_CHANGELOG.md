# Week 7 Phase 5 Changelog — FastAPI + RDS Auth Foundation

## Added

- `api/main.py` FastAPI application.
- `/health`, `/health/artifacts`, `/auth/login`, `/auth/me`, `/claims/validate`, `/claims/recommend` endpoints.
- Dependency-light JWT-like HS256 token creation/verification.
- PBKDF2 password hashing without passlib dependency.
- `AuthRepository` supporting local SQLite and AWS RDS PostgreSQL via optional `psycopg`.
- Auth DB initialization script.
- API preflight script.
- Auth/security unit tests.
- FastAPI import smoke test.

## Changed

- Moved OpenAI SDK out of base `requirements.txt` into `requirements-openai.txt` so corporate machines can install the base project.
- Added `requirements-api.txt` for FastAPI/RDS dependencies.
- Expanded `.env.example` for API/auth/RDS/OpenAI/AWS-compatible config.

## Design note

OpenAI remains the final presentation layer only. The structured facts come from validation, ML prediction, SHAP/business reasons, RAG evidence, and deterministic recommendations.
