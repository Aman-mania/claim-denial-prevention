# Dashboard Import Fix

## Problem
`dev_dashboard/app.py` imports:

```python
from tabs.policy_rag import render_policy_rag_tab
```

but the previous cleanup patch did not expose `render_policy_rag_tab` from `dev_dashboard/tabs/policy_rag.py`, causing Streamlit to fail at startup.

## Fix
- Replaced `dev_dashboard/tabs/policy_rag.py` with a robust implementation that defines `render_policy_rag_tab()`.
- Added backward-compatible aliases:
  - `render_policy_evidence_tab`
  - `render_policy_tab`
- Added a dashboard import smoke-check tool.
- Added a test that verifies the expected function is exported.

## Verification

```bash
python -m py_compile dev_dashboard/tabs/policy_rag.py tools/check_dashboard_imports.py
python tools/check_dashboard_imports.py
pytest tests/dashboard/test_policy_rag_import.py -v
streamlit run dev_dashboard/app.py
```
