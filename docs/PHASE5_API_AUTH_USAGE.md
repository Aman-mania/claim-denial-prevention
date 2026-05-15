# Phase 5 — FastAPI + Auth/RBAC Usage

## Install base requirements without OpenAI

```bash
python -m pip install -r requirements.txt
```

OpenAI is optional and intentionally moved to `requirements-openai.txt` because some corporate networks block the package.

## Test non-OpenAI Phase 1–4 pieces

```bash
python -m py_compile src/rules/claim_validator.py src/agent/remediation_agent.py src/agent/openai_output.py
pytest tests/rules tests/agent -v
```

The OpenAI layer will automatically use deterministic fallback when `ENABLE_OPENAI_AGENT_OUTPUT=false` or the `openai` package is unavailable.

## Install API dependencies when network allows

```bash
python -m pip install -r requirements-api.txt
```

## Initialize local auth DB

```bash
python scripts/init_auth_db.py \
  --database-url sqlite:///data/auth/auth.db \
  --developer-email developer@example.com --developer-password dev12345 \
  --analyst-email analyst@example.com --analyst-password analyst12345 \
  --overwrite
```

## Start FastAPI locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"analyst@example.com","password":"analyst12345"}'
```

Copy `access_token`.

## Validate claim

```bash
curl -X POST http://localhost:8000/claims/validate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CUSTOM001","patient_id":"P001","provider_id":"PR100","diagnosis_code":"","procedure_code":"PROC1","billed_amount":12000}'
```

## Generate recommendation

Run the usual pipeline first so local artifacts exist:

```bash
python run_gold.py
python run_train.py
python run_explain.py
python run_week6.py --mode preferred
```

Then:

```bash
curl -X POST http://localhost:8000/claims/recommend \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"CUSTOM001","patient_id":"P001","provider_id":"PR100","diagnosis_code":"","procedure_code":"PROC1","billed_amount":12000}'
```

## AWS/RDS mode later

Set:

```env
AUTH_DATABASE_URL=postgresql://claim_app_user:<password>@<rds-endpoint>:5432/claim_denial
JWT_SECRET=<secret-from-secrets-manager>
ENABLE_OPENAI_AGENT_OUTPUT=true
OPENAI_API_KEY=<secret-from-secrets-manager>
```

Then run the same `scripts/init_auth_db.py` command against RDS.
