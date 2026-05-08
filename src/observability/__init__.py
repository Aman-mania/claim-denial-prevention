"""Observability utilities: error codes, tracking, and CloudWatch-ready metrics."""

from src.observability.error_codes import ErrorCategory, ErrorCode, ErrorSeverity
from src.observability.error_tracker import ErrorEvent, ErrorTracker
from src.observability.exceptions import ClaimDenialError
from src.observability.pipeline_integration import (
    record_exception_and_return_report,
    record_pipeline_report,
    summarize_error_events,
    tracker_from_env,
)

__all__ = [
    "ClaimDenialError",
    "ErrorCategory",
    "ErrorCode",
    "ErrorEvent",
    "ErrorSeverity",
    "ErrorTracker",
    "record_exception_and_return_report",
    "record_pipeline_report",
    "summarize_error_events",
    "tracker_from_env",
]
