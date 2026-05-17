# Phase 8 — Local Docker + AWS-ready deployment

## Goal

Phase 8 makes the project run like a deployable product while still keeping local development simple.

The same code path is used for local, Docker, and AWS. Environment differences are controlled centrally through `APP_ENV` and related environment variables.

```text
APP_ENV=local  -> Python venv + SQLite + localhost API
APP_ENV=docker -> Docker Compose + SQLite + service-name networking
APP_ENV=aws    -> EC2 Docker Compose + RDS PostgreSQL + S3/Secrets values
```

## Why central runtime config instead of many if/else checks?

Do not scatter `if local else aws` across ML, RAG, API, and UI modules. That becomes hard to test and easy to break.

Instead, this patch adds:

```text
src/config/runtime.py
```

All environment-specific differences are resolved there. The rest of the app receives normal values like:

```text
AUTH_DATABASE_URL
CLAIM_DENIAL_API_BASE_URL
GOLD_DIR
MODELS_DIR
RAG_VECTOR_DIR
ENABLE_OPENAI_AGENT_OUTPUT
```

## Files added

```text
Dockerfile.api
Dockerfile.streamlit
docker-compose.yml
.dockerignore
.env.local.example
.env.docker.example
.env.aws.example
requirements-aws.txt
src/config/runtime.py
scripts/check_phase8_deployment.py
scripts/docker_smoke_test.sh
scripts/sync_artifacts_to_s3.py
```

## Local Docker test

1. Make sure your normal pipeline artifacts exist:

```bash
python run_gold.py
python run_train.py
python run_explain.py
python run_week6.py --mode fallback
```

2. Copy Docker env:

```bash
cp .env.docker.example .env.docker
```

3. Initialize local auth DB if not already done:

```bash
python scripts/init_auth_db.py \
  --database-url sqlite:///data/auth/auth.db \
  --developer-email developer@example.com \
  --developer-password dev12345 \
  --analyst-email analyst@example.com \
  --analyst-password analyst12345 \
  --overwrite
```

4. Build and start:

```bash
docker compose build
docker compose up
```

5. Open:

```text
FastAPI docs: http://localhost:8000/docs
Product UI:    http://localhost:8501
```

6. In another terminal, smoke test:

```bash
bash scripts/docker_smoke_test.sh
```

## AWS-ready mode

For EC2 deployment, use the same Docker Compose pattern, but copy `.env.aws.example` to `.env.docker` on the EC2 instance and replace:

```text
AUTH_DATABASE_URL=postgresql://...
JWT_SECRET=...
OPENAI_API_KEY=...
S3_BUCKET_NAME=...
API_CORS_ORIGINS=...
```

The Docker Compose service names remain the same. The Streamlit UI can still call the API at:

```text
http://api:8000
```

inside the Docker network.

## Artifact handling

First AWS deployment can use local EC2 files mounted into Docker containers:

```text
./data:/app/data
./models:/app/models
./logs:/app/logs
```

For backup and later repeatable deploys, use:

```bash
python -m pip install -r requirements-aws.txt
python scripts/sync_artifacts_to_s3.py upload --bucket <bucket> --prefix claim-denial/dev
```

On EC2:

```bash
python scripts/sync_artifacts_to_s3.py download --bucket <bucket> --prefix claim-denial/dev
```

## Environment decision matrix

| Concern | Local | Docker local | AWS |
|---|---|---|---|
| APP_ENV | local | docker | aws |
| Auth DB | SQLite | SQLite | RDS PostgreSQL |
| API URL for UI | localhost | http://api:8000 | http://api:8000 inside Compose |
| Artifacts | local files | mounted local files | EC2 files restored from S3 |
| OpenAI | disabled | disabled by default | enabled after key configured |
| RAG | TF-IDF fallback recommended for corporate machines | fallback by default | preferred semantic if model/cache available |

## Common errors

### API is healthy but UI cannot login

Check `CLAIM_DENIAL_API_BASE_URL` in `.env.docker`. In Docker Compose it should be:

```text
http://api:8000
```

### AWS starts with SQLite

That should not happen. `APP_ENV=aws` requires a PostgreSQL `AUTH_DATABASE_URL` and raises an error if missing.

### RAG tries to call Hugging Face on corporate network

Use fallback mode locally:

```text
RAG_MODE=fallback
RAG_EMBEDDING_BACKEND=tfidf
RAG_VECTOR_BACKEND=numpy
```

or cache the model and set offline mode.
