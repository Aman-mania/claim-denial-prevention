"""
Observability constants.

These are intentionally separate from src/constants.py so error handling can be
applied without disturbing existing project constants.
"""

from pathlib import Path

DEFAULT_LOG_DIR = Path("logs")
ERROR_EVENTS_FILENAME = "error_events.jsonl"
ERROR_SUMMARY_FILENAME = "error_summary.json"
ERROR_METRICS_NAMESPACE = "ClaimDenialPrevention"
ERROR_REPEAT_ALERT_THRESHOLD = 2

# Keep dimensions low-cardinality for CloudWatch cost/control.
EMF_DIMENSIONS = ("component", "stage", "error_code", "severity")

# Metadata keys used to group repeated errors. Avoid claim_id/request_id here.
ERROR_FINGERPRINT_FIELDS = (
    "stage",
    "dataset",
    "path",
    "file",
    "table",
    "model_name",
    "feature",
)
