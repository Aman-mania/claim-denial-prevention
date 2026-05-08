# Observability Integration Update

## What changed

This patch wires the structured error-tracking layer into the rest of the local pipeline without overwriting the fast-moving Week 4 code.

Added:

- `src/observability/pipeline_integration.py`
  - Creates `ErrorTracker` from environment variables.
  - Converts pipeline reports into structured error events.
  - Records dataset-level failures, schema validation warnings, Gold/ML failures, repeated occurrences.
- `tools/apply_observability_integration.py`
  - Idempotently patches `run_ingestion.py`, `run_silver.py`, `run_gold.py`, and `run_train.py`.
  - Preserves your current optimized code instead of replacing whole files.
- `tests/observability/test_pipeline_integration.py`
  - Tests failed dataset reports, repeated-error detection, validation warnings, and summary output.

## How to apply

```bash
unzip claim_denial_observability_pipeline_integration.zip -d /tmp/obs_patch
rsync -av /tmp/obs_patch/ ./
python tools/apply_observability_integration.py
```

Then run your normal pipeline:

```bash
python run_ingestion.py
python run_silver.py
python run_gold.py
python run_train.py
python run_error_report.py
```

## Environment variables

```bash
export ERROR_LOG_DIR=logs
export ERROR_REPEAT_THRESHOLD=2
export ERROR_EMIT_EMF=false
```

For AWS/ECS/Lambda/Databricks jobs, set `ERROR_EMIT_EMF=true` to print CloudWatch EMF-compatible metric JSON to stdout.

## Why this design

The project currently changes quickly across local Week 4 iterations. Instead of replacing entire pipeline classes and risking loss of your current optimized model code, this patch instruments the stable entry points and centralizes report parsing in one reusable helper. Later, the same helper can be called from Databricks notebooks/jobs or FastAPI routes.
