# Stabilization Before Week 5/6

This patch implements the stabilization recommendations before RAG/agent work.

## Fixes

- Fixes `ClaimDenialService.load()` by using `CustomClaimFeatureBuilder`.
- Adds `ClaimFeatureBuilder` alias for backward compatibility.
- Adds `build_features()` method on the custom feature builder.
- Updates dashboard ML tab to:
  - use tuned thresholds from `training_report.json`;
  - compute confusion matrices using tuned threshold instead of default `.predict()`;
  - call `ClaimDenialService` for custom raw-claim inference;
  - remove the old feature-toggle custom builder flow;
  - show review/denial thresholds in prediction banners;
  - display calibration metrics from the new report structure.
- Updates SHAP labels and fix suggestions for current Gold features:
  - `billed_deviation_imputed_capped`
  - `billed_amount_imputed`
  - `log_billed_amount_imputed`
  - `cost_match_encoded`
  - `severity_rank`
- Corrects SHAP `model_output` to include `expected_value + contribution_sum`.
- Wires observability report recording into:
  - `run_ingestion.py`
  - `run_silver.py`
  - `run_gold.py`
  - `run_train.py`
- Updates README to reflect current project status.
- Adds tests for:
  - custom claim service success/error envelopes;
  - explainability labels for current feature names.

## Apply

```bash
unzip claim_denial_stabilization_before_week5_6.zip -d /tmp/stabilize_patch
rsync -av /tmp/stabilize_patch/ ./
```

## Verify

```bash
python -m py_compile run_ingestion.py run_silver.py run_gold.py run_train.py
pytest tests/inference/test_feature_builder.py tests/inference/test_claim_service.py tests/ml/test_explain_labels.py tests/observability/test_error_tracker.py
python run_ingestion.py
python run_silver.py
python run_gold.py
python run_train.py
streamlit run dev_dashboard/app.py
```
