# Week 5 Parquet Type Fix

## Problem

`python run_explain.py` generated all claim explanations successfully but failed while writing
`gold_claim_explanations.parquet` with PyArrow:

```text
Could not convert 112.75 with type float: tried to convert to boolean
Conversion failed for column feature_value with type object
```

The `feature_value` column stores evidence values from different source features. Some rows
contain booleans (`True` for missing flags), while other rows contain floats (`112.75` for
billing deviation), strings, or nulls. Parquet needs a stable physical type per column.

## Fix

- Serialize `feature_value` as a nullable string before writing Parquet.
- Keep numeric columns numeric: `risk_score`, thresholds, and `shap_value` remain floats.
- Keep integer-like columns as nullable `Int64`: `predicted_denial`, `reason_rank`.
- Add schema coercion for both long explanation output and summary output before writing.
- Preserve XAI-specific error classification for write failures by raising `ClaimDenialError`
  with `XAI_EXPLANATION_WRITE_FAILED` instead of wrapping the issue as `XAI_999`.
- Add a regression test that mixes `True` and `112.75` in `feature_value`, writes Parquet,
  and reads it back successfully.

## Files changed

```text
src/explainability/explanation_generator.py
src/explainability/schemas.py
tests/explainability/test_explanation_outputs.py
```

## Verify

```bash
python -m py_compile src/explainability/explanation_generator.py
pytest tests/explainability/test_explanation_outputs.py -v
python run_explain.py
```
