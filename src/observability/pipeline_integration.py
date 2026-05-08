"""
Pipeline observability integration helpers.

These functions wire the central error catalog + ErrorTracker into the existing
local entry points without forcing large rewrites of the Bronze/Silver/Gold/ML
classes. They are intentionally small and pure so they can also be called from
Databricks jobs, FastAPI handlers, ECS tasks, or Lambda functions later.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

from src.observability.error_codes import ErrorCode, ErrorSeverity
from src.observability.error_tracker import ErrorEvent, ErrorTracker

logger = structlog.get_logger(__name__)


_COMPONENT_FAILURE_CODE: dict[str, ErrorCode] = {
    "ingestion": ErrorCode.INGEST_UNEXPECTED,
    "bronze": ErrorCode.INGEST_UNEXPECTED,
    "silver": ErrorCode.SILVER_UNEXPECTED,
    "gold": ErrorCode.GOLD_UNEXPECTED,
    "ml": ErrorCode.ML_UNEXPECTED,
    "training": ErrorCode.ML_UNEXPECTED,
    "inference": ErrorCode.INFER_UNEXPECTED,
    "dashboard": ErrorCode.INFER_UNEXPECTED,
}

_DATASET_FAILURE_CODE: dict[tuple[str, str], ErrorCode] = {
    ("ingestion", "missing_file"): ErrorCode.INGEST_RAW_FILE_MISSING,
    ("ingestion", "unknown_dataset"): ErrorCode.INGEST_UNKNOWN_DATASET,
    ("silver", "missing_file"): ErrorCode.SILVER_BRONZE_FILE_MISSING,
    ("gold", "missing_file"): ErrorCode.GOLD_SILVER_FILE_MISSING,
    ("ml", "missing_file"): ErrorCode.ML_GOLD_FEATURES_MISSING,
}

_VALIDATION_WARNING_CODE: dict[str, ErrorCode] = {
    "ingestion": ErrorCode.INGEST_SCHEMA_WARNING,
    "bronze": ErrorCode.INGEST_SCHEMA_WARNING,
    "silver": ErrorCode.SILVER_VALIDATION_WARNING,
}


def tracker_from_env(*, default_log_dir: str | Path = "logs") -> ErrorTracker:
    """
    Create an ErrorTracker configured through environment variables.

    Environment variables
    ---------------------
    ERROR_LOG_DIR: where JSONL/summary files are written; default logs/
    ERROR_EMIT_EMF: true/1/yes enables CloudWatch EMF JSON on stdout
    ERROR_REPEAT_THRESHOLD: occurrence count at which is_repeated becomes true
    """
    log_dir = Path(os.getenv("ERROR_LOG_DIR", str(default_log_dir)))
    emit_emf = os.getenv("ERROR_EMIT_EMF", "false").strip().lower() in {"1", "true", "yes", "y"}
    try:
        repeat_threshold = int(os.getenv("ERROR_REPEAT_THRESHOLD", "2"))
    except ValueError:
        repeat_threshold = 2
    return ErrorTracker(log_dir=log_dir, emit_emf=emit_emf, repeat_alert_threshold=repeat_threshold)


def classify_failure(component: str, error_message: str | None = None) -> ErrorCode:
    """Best-effort mapping from a textual failure to a stable ErrorCode."""
    component = component.lower()
    msg = (error_message or "").lower()
    if "not found" in msg or "missing" in msg or "no such file" in msg:
        return _DATASET_FAILURE_CODE.get((component, "missing_file"), _COMPONENT_FAILURE_CODE.get(component, ErrorCode.SYSTEM_UNEXPECTED))
    if "unknown dataset" in msg or "not in" in msg and "registry" in msg:
        return _DATASET_FAILURE_CODE.get((component, "unknown_dataset"), _COMPONENT_FAILURE_CODE.get(component, ErrorCode.SYSTEM_UNEXPECTED))
    if "row count" in msg or "duplicat" in msg and component == "gold":
        return ErrorCode.GOLD_ROW_COUNT_MISMATCH
    if "threshold" in msg and component in {"ml", "training"}:
        return ErrorCode.ML_THRESHOLD_TUNING_FAILED
    if "calibration" in msg and component in {"ml", "training"}:
        return ErrorCode.ML_CALIBRATION_FAILED
    return _COMPONENT_FAILURE_CODE.get(component, ErrorCode.SYSTEM_UNEXPECTED)


def record_validation_warnings(
    validation: dict[str, Any] | None,
    *,
    tracker: ErrorTracker,
    component: str,
    stage: str,
    dataset: str | None = None,
) -> list[ErrorEvent]:
    """Record schema validation warnings returned by Bronze/Silver reports."""
    events: list[ErrorEvent] = []
    if not validation:
        return events
    status = str(validation.get("status", "")).lower()
    if status != "warnings":
        return events
    code = _VALIDATION_WARNING_CODE.get(component.lower(), ErrorCode.SYSTEM_UNEXPECTED)
    errors = validation.get("errors")
    failure_count = len(errors) if isinstance(errors, list) else None
    events.append(
        tracker.record(
            code,
            f"{component.title()} validation produced warnings"
            + (f" for dataset '{dataset}'" if dataset else ""),
            component=component,
            stage=stage,
            severity=ErrorSeverity.WARNING,
            metadata={
                "stage": stage,
                "dataset": dataset,
                "failure_count": failure_count,
            },
        )
    )
    return events


def record_pipeline_report(
    report: dict[str, Any] | None,
    *,
    component: str,
    tracker: ErrorTracker | None = None,
    stage: str = "pipeline_run",
) -> list[ErrorEvent]:
    """
    Inspect a pipeline report and record structured errors/warnings.

    Supports current report shapes:
    - Bronze/Silver: {"datasets": {name: {"status": ..., "validation": ...}}}
    - Gold/ML: {"status": "success"|"failed", "error": ...}
    """
    tracker = tracker or tracker_from_env()
    events: list[ErrorEvent] = []

    if not isinstance(report, dict):
        events.append(
            tracker.record(
                _COMPONENT_FAILURE_CODE.get(component.lower(), ErrorCode.SYSTEM_UNEXPECTED),
                f"{component} returned no structured report.",
                component=component,
                stage=stage,
                metadata={"stage": stage},
            )
        )
        return events

    datasets = report.get("datasets")
    if isinstance(datasets, dict):
        for dataset, result in datasets.items():
            if not isinstance(result, dict):
                continue
            result_status = str(result.get("status", "")).lower()
            dataset_stage = f"{stage}.{dataset}"

            events.extend(
                record_validation_warnings(
                    result.get("validation"),
                    tracker=tracker,
                    component=component,
                    stage=dataset_stage,
                    dataset=str(dataset),
                )
            )

            if result_status == "failed":
                error_message = str(result.get("error") or f"{component} failed for dataset {dataset}")
                code = classify_failure(component, error_message)
                events.append(
                    tracker.record(
                        code,
                        error_message,
                        component=component,
                        stage=dataset_stage,
                        metadata={
                            "stage": dataset_stage,
                            "dataset": str(dataset),
                            "status": result_status,
                        },
                    )
                )
        return events

    status = str(report.get("status", "success")).lower()
    if status not in {"success", "ok", "passed"}:
        error_message = str(report.get("error") or f"{component} pipeline failed")
        code = classify_failure(component, error_message)
        events.append(
            tracker.record(
                code,
                error_message,
                component=component,
                stage=stage,
                metadata={
                    "stage": stage,
                    "status": status,
                    "recommended_model": report.get("recommended"),
                },
            )
        )
    return events


def record_exception_and_return_report(
    exc: BaseException,
    *,
    component: str,
    stage: str,
    tracker: ErrorTracker | None = None,
    fallback_code: ErrorCode | None = None,
) -> dict[str, Any]:
    """Record an exception and return a pipeline-style failure report."""
    tracker = tracker or tracker_from_env()
    event = tracker.record_exception(
        exc,
        component=component,
        stage=stage,
        fallback_code=fallback_code or _COMPONENT_FAILURE_CODE.get(component.lower(), ErrorCode.SYSTEM_UNEXPECTED),
        metadata={"stage": stage},
    )
    return {
        "status": "failed",
        "error": str(exc),
        "error_code": event.error_code,
        "error_event_id": event.event_id,
        "occurrence_count": event.occurrence_count,
        "is_repeated": event.is_repeated,
    }


def summarize_error_events(events: list[ErrorEvent]) -> dict[str, Any]:
    """Small summary suitable for run scripts / notebooks."""
    if not events:
        return {"errors_recorded": 0, "repeated_errors": 0, "codes": []}
    return {
        "errors_recorded": len(events),
        "repeated_errors": sum(1 for e in events if e.is_repeated),
        "codes": sorted({e.error_code for e in events}),
    }
