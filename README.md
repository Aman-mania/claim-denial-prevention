# AI-Powered Claim Denial Prevention & Remediation System

> Real-time decision support for validating healthcare claims **before** submission.  
> Predicts denial risk · Explains reasons · Suggests fixes · Tracks operational errors

---

## Project status

| Week | Module | Status |
|------|--------|--------|
| 1 | Bronze ingestion pipeline | ✅ Complete |
| 2 | Analytics layer + Streamlit dashboard basics | ✅ Mostly complete |
| 3 | Silver layer cleaning | ✅ Complete |
| 4 | Gold feature engineering + ML training | ✅ Complete / stabilization ongoing |
| 5 | Explainability / SHAP reason layer | 🟡 Partially implemented |
| 6 | RAG policy retrieval | ⏳ Not started |
| 7 | Decision engine + FastAPI + Auth | ⏳ Not started |
| 8 | AWS / Databricks deployment | ⏳ Not started |

---

## Current pipeline

```text
Raw CSVs
  → Bronze ingestion + profiling
  → Silver cleaning + data quality flags
  → Gold feature engineering + inference artifacts
  → ML training: Logistic Regression + XGBoost
  → Threshold tuning + calibration report + model card
  → Prediction + SHAP reason codes
  → Custom claim scoring service
  → Streamlit development dashboard
```

---

## Data files

Place these files in `data/raw/`:

```bash
claims_1000.csv
providers_1000.csv
diagnosis.csv
cost.csv
```

The current replacement claims dataset may include a real `denial_flag`. When present, Gold uses it as the supervised ML target. If it is absent, Gold falls back to a documented synthetic-label path for legacy data.

---

## Run locally

```bash
python run_ingestion.py
python run_analytics.py
python run_silver.py
python run_gold.py
python run_train.py
streamlit run dev_dashboard/app.py
```

Run tests:

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

Check repeated operational errors:

```bash
python run_error_report.py
```

---

## Layer responsibilities

### Bronze

- Loads raw CSV files.
- Performs soft Pandera validation.
- Adds metadata: `ingestion_timestamp`, `source_file`.
- Writes immutable Bronze Parquet.
- Does **not** clean, impute, or drop data.

### Silver

- Normalizes codes/text.
- Parses dates.
- Preserves real `denial_flag` if available.
- Keeps `billed_amount` null when missing.
- Adds missing/logic flags:
  - `diagnosis_code_missing`
  - `procedure_code_missing`
  - `billed_amount_missing`
  - `proc_no_diag`
  - `diag_no_proc`

### Gold

- Joins claims with provider, diagnosis, and cost references.
- Preserves one row per claim.
- Uses regional cost match first, procedure-level fallback second.
- Preserves raw `billed_amount`.
- Adds model-ready imputed amount features:
  - `billed_amount_imputed`
  - `amount_imputation_strategy`
  - `log_billed_amount_imputed`
  - `billed_deviation_imputed_capped`
- Writes:
  - `gold_claim_base.parquet`
  - `gold_claim_features.parquet`
  - `feature_manifest.json`
  - `inference_artifacts.json`

### ML

- Trains Logistic Regression and XGBoost.
- Uses train/validation/test split.
- Tunes decision threshold on validation data instead of hardcoding `0.50`.
- Reports final metrics on held-out test data.
- Saves:
  - `lr_model.pkl`
  - `xgb_model.pkl`
  - `training_report.json`
  - `threshold_report.json`
  - `calibration_report.json`
  - `model_card.json`

### Inference

Use `ClaimDenialService` for custom claim scoring. The dashboard/API should send raw claim fields, not manual ML feature toggles.

```python
from pathlib import Path
from src.inference.claim_service import ClaimDenialService

service = ClaimDenialService.load(
    gold_dir=Path("data/gold"),
    models_dir=Path("models"),
)

result = service.score_claim({
    "claim_id": "CUSTOM001",
    "patient_id": "P001",
    "provider_id": "PR100",
    "diagnosis_code": "D10",
    "procedure_code": "PROC1",
    "billed_amount": 12000,
})
```

---

## Error handling and observability

The project has a central error-code catalog in `src/observability/error_codes.py`.

Examples:

```text
INGEST_001  raw input file missing
SILVER_001  Bronze file missing for Silver
GOLD_003    Gold row count mismatch
ML_005      threshold tuning failed
INFER_006   inconsistent custom claim state
```

Errors are recorded to:

```text
logs/error_events.jsonl
logs/error_summary.json
```

The summary file tracks whether an error has occurred more than once.

---

## Design principles

- **Bronze is immutable** — never clean in Bronze.
- **Silver is auditable** — preserve null financial fields and add flags.
- **Gold is model-ready** — add imputed/encoded features without overwriting raw values.
- **Thresholds are tuned** — do not rely on a blind `0.50` decision boundary.
- **Custom inference uses raw claims** — avoid training-serving skew.
- **Stable error codes** — all repeated issues should be searchable and countable.
- **Local-first, cloud-ready** — current files can later move to S3/Delta/Databricks/MLflow with minimal interface changes.

---

## Next implementation phase

Before Week 6 RAG, stabilize Week 5 explainability:

1. Confirm dashboard uses `ClaimDenialService`.
2. Confirm SHAP labels match all current ML features.
3. Add end-to-end smoke tests.
4. Add policy-document loader and chunking interfaces.
5. Implement local FAISS RAG, then keep the interface swappable for Databricks/AWS.
