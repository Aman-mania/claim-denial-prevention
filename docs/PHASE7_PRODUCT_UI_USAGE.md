# Phase 7 Product UI Usage

## Purpose

The Phase 7 product UI is a single Streamlit application with role-aware access.
It does not replace the existing development dashboard. The old dashboard remains useful for technical inspection:

```bash
streamlit run dev_dashboard/app.py
```

The product UI is for the final integrated workflow:

```bash
streamlit run product_ui/app.py
```

## Roles

### Analyst

Analyst users see only business-facing sections:

- Overview
- Claim Analytics
- Custom Claim
- System Health

The Custom Claim tab calls FastAPI:

```text
Streamlit → FastAPI → validation → ML → SHAP/reason mapping → RAG → agent → response
```

### Developer

Developer users see technical sections:

- Overview
- Custom Claim
- Data Pipeline
- Risk Model
- Risk Explanations
- Policy Evidence
- Retrieval Analytics
- System Health

## Start locally

Start the API first:

```bash
export AUTH_DATABASE_URL=sqlite:///data/auth/auth.db
export JWT_SECRET=local-dev-secret-change-later
export ENABLE_OPENAI_AGENT_OUTPUT=false
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then start the product UI:

```bash
streamlit run product_ui/app.py
```

Login with:

```text
analyst@example.com / analyst12345
developer@example.com / dev12345
```

## Low-risk behavior

LOW-risk claims no longer show confusing denial-risk reason cards based on tiny SHAP values. They can still show validation warnings and basic action-plan notes.

## AWS migration note

For local development, SQLite remains sufficient:

```env
AUTH_DATABASE_URL=sqlite:///data/auth/auth.db
```

For AWS, replace it with RDS PostgreSQL:

```env
AUTH_DATABASE_URL=postgresql://<user>:<password>@<rds-endpoint>:5432/<db>
```
