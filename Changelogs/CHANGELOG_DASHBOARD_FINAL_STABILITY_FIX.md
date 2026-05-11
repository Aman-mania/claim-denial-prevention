# Dashboard Final Stability Fix

Fixes dashboard issues observed after tab cleanup:

- `tools/check_dashboard_imports.py` now adds both repo root and `dev_dashboard/` to `sys.path`, so imports like `src.constants` and `tabs.*` both resolve.
- `Risk Explanations` and `Policy Evidence` claim selectors now use unique Streamlit widget keys:
  - `risk_explanations_claim_select`
  - `policy_evidence_claim_select`
- Updated the two patched tabs to use `width="stretch"` instead of deprecated `use_container_width=True`.
- Added `tools/apply_dashboard_global_hardening.py` to replace deprecated width arguments across the entire dashboard.
- Added runtime contract tests for dashboard imports, renderer signatures, selectbox keys, and deprecated width usage in the updated tabs.

Recommended verification:

```bash
python tools/apply_dashboard_global_hardening.py
python -m py_compile dev_dashboard/tabs/policy_rag.py dev_dashboard/tabs/explainability.py tools/check_dashboard_imports.py tools/apply_dashboard_global_hardening.py
python tools/check_dashboard_imports.py
pytest tests/dashboard/test_dashboard_runtime_contracts.py -v
streamlit run dev_dashboard/app.py
```
