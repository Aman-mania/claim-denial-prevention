# Week 5 hardening before Week 6 RAG

## Added
- On-demand `ExplanationService` for custom claims and future FastAPI.
- Critical-rule-first reason mapping so missing diagnosis/procedure/amount reasons are never hidden by SHAP ranking.
- XAI-specific error codes and Week 6 RAG placeholder error codes in the central error catalog.
- Direct `ErrorTracker` integration in the Week 5 explanation pipeline.
- Modular pytest markers and `tools/run_tests.py` for grouped test execution.
- Additional Week 5 tests for catalog, mapper, service, error handling, and XAI error-code registration.

## Changed
- Week 5 batch generation now explicitly explains XGBoost predictions with the XGBoost SHAP explainer to avoid prediction/explanation model mismatch.
- Explanation schema now includes `evidence_type` to distinguish `critical_rule`, `shap`, and `fallback` reasons.
- `run_explain.py` now reports `success_with_warnings` when some claims fail and points to error logs.
- Gold feature input is read through the table-store abstraction, preparing for Delta/Unity Catalog replacement later.

## Why
Week 6 RAG should consume stable reason codes, policy queries, and policy tags. This hardening patch makes Week 5 outputs more deterministic, observable, testable, and cloud-ready before policy retrieval is added.
