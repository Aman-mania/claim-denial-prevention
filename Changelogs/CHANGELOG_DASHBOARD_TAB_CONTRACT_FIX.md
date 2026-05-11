# Dashboard Tab Contract Fix

## Problem

`dev_dashboard/app.py` calls tab renderers with shared keyword arguments such as
`root_dir`, `gold_dir`, and `models_dir`. Previous dashboard UI cleanup files did
not preserve this renderer contract, causing Streamlit failures such as:

- `ImportError: cannot import name 'render_policy_rag_tab'`
- `TypeError: render_policy_rag_tab() got an unexpected keyword argument 'root_dir'`

The same issue could affect the Risk Explanations tab.

## Fix

- Replaced `dev_dashboard/tabs/policy_rag.py` with a stable `render_policy_rag_tab(root_dir=None, gold_dir=None, models_dir=None, **kwargs)` function.
- Replaced `dev_dashboard/tabs/explainability.py` with a stable `render_explainability_tab(root_dir=None, gold_dir=None, models_dir=None, **kwargs)` function.
- Added backward-compatible aliases:
  - `render_policy_evidence_tab`
  - `render_policy_tab`
  - `render_risk_explanations_tab`
  - `render_explanation_tab`
- Added `tools/check_dashboard_imports.py` to smoke-check dashboard imports without launching Streamlit.
- Added tests under `tests/dashboard/test_dashboard_tab_contracts.py`.

## UI behavior preserved

- Functionality-focused labels.
- No visible Week 5 / Week 6 wording inside the tabs.
- Reduced emojis and presentation-noise.
- Deduplicated policy evidence display.
- Generated narrative moved into a collapsed expander.
- Structured reasons and policy evidence shown once.

## Verify

```bash
python -m py_compile dev_dashboard/tabs/policy_rag.py dev_dashboard/tabs/explainability.py tools/check_dashboard_imports.py
python tools/check_dashboard_imports.py
pytest tests/dashboard/test_dashboard_tab_contracts.py -v
streamlit run dev_dashboard/app.py
```
