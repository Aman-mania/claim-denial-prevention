# Error Handling + Error Codes Patch

## What changed

This patch adds a structured, AWS-ready error handling layer without disturbing the existing medallion pipeline.

### New code

- `src/observability/error_codes.py`
  - Central `ErrorCode` enum
  - `ErrorSeverity`, `ErrorCategory`
  - `ERROR_DEFINITIONS` catalog with retryability and user-facing messages

- `src/observability/error_tracker.py`
  - Writes every error occurrence to `logs/error_events.jsonl`
  - Maintains aggregated counts in `logs/error_summary.json`
  - Marks `is_repeated=True` once the same error fingerprint reaches count >= 2

- `src/observability/emf.py`
  - Builds CloudWatch Embedded Metric Format JSON for future AWS/ECS/Lambda/CloudWatch use

- `src/observability/exceptions.py`
  - Adds `ClaimDenialError`, a base application exception carrying a stable error code

- `src/inference/claim_service.py`
  - Safe custom-claim scoring wrapper
  - Validates raw claim input
  - Detects contradictory custom-builder states
  - Returns structured error envelopes instead of crashing

- `tools/error_report.py` / `run_error_report.py`
  - CLI report for repeated errors

- `tests/observability/test_error_tracker.py`
  - Unit tests for repeated error counts and error-code preservation

## Local usage

```bash
python run_error_report.py
python run_error_report.py --min-count 1
python run_error_report.py --json
```

## Example error response from inference

```json
{
  "status": "error",
  "error": {
    "error_code": "INFER_006",
    "message": "Inconsistent claim: billed_amount_missing=True while billed_amount is present.",
    "severity": "WARNING",
    "category": "data_quality",
    "retryable": false,
    "occurrence_count": 2,
    "is_repeated": true
  },
  "prediction": null,
  "features": null
}
```

## Why this matters for AWS migration

The local tracker uses JSONL and JSON summary files. This keeps development simple, but the event shape is already structured for:

- CloudWatch Logs
- CloudWatch Embedded Metric Format metrics
- Databricks job logs
- S3 log archiving
- future FastAPI error responses

The error code is stable, while error messages can evolve.
