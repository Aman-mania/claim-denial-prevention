# Week 7 Phase 1-4 Status Update

This patch prepares the project for the Week 7 integrated product layer.

## Implemented locally so far

- Bronze ingestion
- Silver cleaning
- Gold feature engineering
- ML training and threshold tuning
- Custom claim scoring service
- SHAP/business explanations
- Policy RAG ingestion and retrieval
- Retrieval analytics dashboard
- Lightweight custom-claim validation
- Deterministic remediation recommendation engine
- Optional OpenAI presentation layer

## Not implemented yet

- FastAPI backend routes
- RDS PostgreSQL auth/RBAC tables
- JWT login flow
- Single role-aware Streamlit product UI
- Docker Compose deployment
- AWS EC2/RDS/S3/CloudWatch/Secrets Manager deployment
- MLflow tracking after AWS deployment

## Intended next phase

Expose `src.agent.remediation_agent.RemediationAgent` through FastAPI:

```python
from pathlib import Path
from src.agent.remediation_agent import RemediationAgent

agent = RemediationAgent.load(
    gold_dir=Path("data/gold"),
    models_dir=Path("models"),
    vector_dir=Path("data/vector_store"),
)

result = agent.analyze_claim({
    "claim_id": "CUSTOM001",
    "patient_id": "P001",
    "provider_id": "PR100",
    "diagnosis_code": "D10",
    "procedure_code": "PROC1",
    "billed_amount": 12000,
})
```

OpenAI can be enabled as the final readable presentation layer using:

```env
ENABLE_OPENAI_AGENT_OUTPUT=true
OPENAI_API_KEY=...
OPENAI_AGENT_MODEL=gpt-4o-mini
```

The deterministic structured fields remain the source of truth even when OpenAI is enabled.
