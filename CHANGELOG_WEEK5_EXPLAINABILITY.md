# Week 5 Explainable AI Implementation

## Added

- `src/explainability/`
  - business reason catalog
  - SHAP-to-reason mapper
  - explanation generation pipeline
  - stable output schemas

- `src/io/`
  - local Parquet table-store abstraction
  - designed so Databricks/Delta migration later swaps IO implementation instead of pipeline logic

- `run_explain.py`
  - generates `gold_claim_explanations.parquet`
  - generates `gold_claim_explanation_summary.parquet`
  - generates `explanation_report.json`

- `dev_dashboard/tabs/explainability.py`
  - new Week 5 dashboard tab
  - does not modify the existing Week 4 ML tab
  - shows business reasons, SHAP evidence, fixes, and Week 6 policy queries

- Updated `dev_dashboard/app.py`
  - adds separate "Explainable AI (Week 5)" tab
  - adds sidebar control for running `run_explain.py`

- Tests
  - `tests/explainability/test_reason_mapper.py`
  - `tests/explainability/test_explanation_outputs.py`

## Design

```text
Gold features + trained model
→ SHAP explanation
→ business reason mapping
→ explanation table
→ Week 6 policy-query handoff
```

The system does not expose raw ML features to the final user. Raw feature names are retained only as debug/audit columns.

## Cloud / Databricks Readiness

Local output remains Parquet for simplicity, but pipeline logic uses a table-store boundary.
For Databricks migration, replace `LocalTableStore` with a Delta/Unity Catalog implementation and keep
the explainability logic unchanged.
