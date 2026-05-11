# Risk Explanations Alignment Fix

## Summary

Restores the Risk Explanations tab to a cleaner, full-width dashboard layout while preserving the runtime fixes added in the previous dashboard stability patches.

## Fixes

- Removes the overly wide left navigation/detail split that made the selected claim content feel misaligned.
- Uses one compact control row for filters and claim selection.
- Presents claim summary metrics in a full-width row.
- Presents business reasons as full-width bordered cards.
- Keeps policy handoff/debug information collapsed.
- Keeps stable renderer signature: `render_explainability_tab(root_dir=None, gold_dir=None, models_dir=None, **kwargs)`.
- Keeps unique Streamlit widget keys to avoid duplicate widget ID failures.
- Uses `width="stretch"` for dataframe rendering.

## Verification

```bash
python -m py_compile dev_dashboard/tabs/explainability.py
pytest tests/dashboard/test_risk_explanations_alignment_contract.py -v
streamlit run dev_dashboard/app.py
```
