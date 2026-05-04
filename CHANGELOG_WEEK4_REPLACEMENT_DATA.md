# Week 4 Replacement Dataset Code Changes

Prepared for `claim-denial-prevention` to support the replacement CSVs.

## Apply

From the repo root, copy/merge this directory over the repo:

```bash
rsync -av /path/to/claim_denial_new_data_changes/ ./
```

Then place the replacement CSVs in `data/raw/` with the existing filenames:

```text
claims_1000.csv
providers_1000.csv
diagnosis.csv
cost.csv
```

Run the usual commands:

```bash
python run_ingestion.py
python run_silver.py
python run_gold.py
python run_train.py
pytest
```

## Files changed

- `src/ingestion/schema.py`
  - Supports optional `denial_flag` in claims.
  - Supports `Medium` diagnosis severity.

- `src/silver/schema.py`
  - Supports optional `denial_flag` in Silver claims.
  - Supports `Medium` diagnosis severity.

- `src/silver/clean.py`
  - Carries real `denial_flag` forward from replacement data.
  - Keeps raw `billed_amount` nulls unchanged in Silver.
  - Coerces `billed_amount` and `denial_flag` safely.

- `src/constants.py`
  - Adds reusable Gold feature column constants for amount imputation, cost match metadata, and label source.

- `src/gold/features.py`
  - Prevents duplicate claim rows when `cost.csv` has multiple regional rows per procedure.
  - Uses regional cost match first, then procedure-level average fallback.
  - Uses provided `denial_flag` when present; synthetic labels remain as fallback for legacy data.
  - Adds median-imputed amount features without overwriting raw `billed_amount`.
  - Adds `cost_match_level`, `cost_match_encoded`, `amount_imputation_strategy`, and `severity_rank`.

- `src/ml/train.py`
  - Updates fallback ML feature list to match the new Gold feature manifest.

- `tests/gold/test_replacement_data_support.py`
  - Adds focused tests for replacement dataset behavior.

## Smoke-check result from uploaded replacement data

Using a local pandas smoke-check, not full pytest because this sandbox lacks repo runtime dependencies (`pandera`, `pyarrow`, `structlog`):

```text
claims: 5000
base rows after Gold join: 5000
feature rows: 5000
label source: provided for all 5000 rows
denial rate: 42.32%
cost matches: procedure_avg=3019, regional=1678, missing=303
raw billed_amount nulls: 472
billed_amount_imputed nulls: 0
amount imputation strategy: original=4528, global_median=303, procedure_median=169
```

## Notes

- Silver still preserves raw null `billed_amount` values for auditability.
- Median imputation is implemented as Gold model-ready features, which is safer for future AWS/Databricks migration because raw and derived fields remain separate.
- `run_gold.py` now receives report fields for label source and cost match counts through the `GoldFeaturePipeline.run()` return dict, though the current CLI does not print those extra fields yet.
