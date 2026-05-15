# Week 7 Phase 7 — Role-aware Product UI + Low-risk Reason Fix

## Summary

This patch adds the single role-aware Streamlit product UI and fixes the low-risk claim issue where tiny SHAP values could create confusing risk reasons for otherwise low-risk claims.

## Key changes

### Agent behavior

- LOW-risk claims no longer generate denial-risk reasons from tiny SHAP contributors.
- LOW-risk claims still show validation warnings and safe recommendations.
- RAG policy retrieval is skipped when there are no risk reasons, reducing unnecessary latency.

### API role filtering

- Developer users continue receiving the full technical response.
- Analyst users receive a business-safe response:
  - no raw feature dictionary;
  - no SHAP debug values;
  - no local source paths;
  - no full reason-definition payload.

### Product UI

New app:

```bash
streamlit run product_ui/app.py
```

The UI calls FastAPI and stores the JWT in Streamlit session state.

Analyst tabs:

- Overview
- Claim Analytics
- Custom Claim
- System Health

Developer tabs:

- Overview
- Custom Claim
- Data Pipeline
- Risk Model
- Risk Explanations
- Policy Evidence
- Retrieval Analytics
- System Health

## Test commands

```bash
python -m py_compile \
  src/agent/remediation_agent.py \
  api/routes/claims.py \
  product_ui/app.py \
  product_ui/api_client.py \
  product_ui/rendering.py \
  tools/check_phase7_product_ui.py

pytest tests/agent/test_low_risk_reason_suppression.py tests/product_ui -v
python tools/run_tests.py --suite week7 --verbose
```

## Manual run

Terminal 1:

```bash
export AUTH_DATABASE_URL=sqlite:///data/auth/auth.db
export JWT_SECRET=local-dev-secret-change-later
export ENABLE_OPENAI_AGENT_OUTPUT=false
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
streamlit run product_ui/app.py
```
