# Week 8 Phase 8 Deployment Changelog

## Added

- Central runtime configuration: `src/config/runtime.py`
- Dockerized FastAPI service: `Dockerfile.api`
- Dockerized Streamlit product UI: `Dockerfile.streamlit`
- Local Docker Compose stack: `docker-compose.yml`
- Environment examples for local, Docker, and AWS
- Phase 8 deployment readiness checker
- Docker/API smoke-test script
- Optional S3 artifact sync helper
- Tests for runtime settings and deployment files

## Behavior

- `APP_ENV=local`: local Python + SQLite + localhost API
- `APP_ENV=docker`: Docker Compose + SQLite + service-name API URL
- `APP_ENV=aws`: EC2 Docker Compose + RDS PostgreSQL required

## Design decision

Environment-specific behavior is centralized in `src/config/runtime.py` instead of scattered if/else logic across the app.
