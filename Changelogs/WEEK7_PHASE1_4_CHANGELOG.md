# Week 7 Phase 1-4 Patch

## Added

- `src/rules`: lightweight custom-claim validation layer.
- `src/agent`: deterministic remediation agent, recommendation catalog, and optional OpenAI presentation layer.
- Tests for validator and recommendation/presentation logic.
- Updated `.env.example` with OpenAI presentation-layer settings and future API/RDS placeholders.
- Updated `tools/run_tests.py` with Week 6/7, RAG, dashboard, rules, and agent suites.

## Design intent

OpenAI is used only as the final presentation layer. It receives already-computed structured facts from validation, ML prediction, explainability, RAG, and deterministic recommendations. If OpenAI is disabled or fails, the system falls back to deterministic readable output.

## AWS readiness

This patch does not require AWS yet. It prepares the core local agent logic so the next phase can expose it via FastAPI and then connect FastAPI to RDS PostgreSQL authentication.
