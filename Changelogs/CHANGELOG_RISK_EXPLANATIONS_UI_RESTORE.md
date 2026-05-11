# Risk Explanations UI Restore

## Why
The previous dashboard hardening patch preserved runtime contracts but simplified the
Risk Explanations tab too much. The result looked like a raw debug list instead of a
professional claim-review dashboard.

## Changes
- Restored the split-pane claim review layout: filters on the left, claim explanation on the right.
- Restored claim labels with claim id, risk level, and risk score.
- Restored metric cards for selected claim risk score, risk level, and prediction.
- Replaced raw bullet-only reasons with bordered reason cards.
- Kept internal week labels out of the UI.
- Kept unique Streamlit widget keys to avoid duplicate element IDs.
- Kept compatibility with `dev_dashboard/app.py` shared renderer arguments.
- Added tests for signature, widget keys, visible week-label removal, and structured layout.

## Verification
```bash
python -m py_compile dev_dashboard/tabs/explainability.py
pytest tests/dashboard/test_risk_explanations_ui_contract.py -v
streamlit run dev_dashboard/app.py
```
