"""
Local + cloud-ready error tracking.

Records every known/unknown error as JSONL and maintains a compact summary file
that answers: "has this error happened more than once?"

Local outputs
-------------
logs/error_events.jsonl   append-only event stream
logs/error_summary.json   count/first_seen/last_seen per fingerprint

Cloud migration
---------------
The same events are structured and can be shipped to CloudWatch Logs, Databricks
logs, or S3 without changing pipeline code. Optional EMF output is available via
emit_emf=True.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.observability.constants import (
    DEFAULT_LOG_DIR,
    ERROR_EVENTS_FILENAME,
    ERROR_FINGERPRINT_FIELDS,
    ERROR_REPEAT_ALERT_THRESHOLD,
    ERROR_SUMMARY_FILENAME,
)
from src.observability.emf import build_error_emf_json
from src.observability.error_codes import (
    ErrorCode,
    ErrorSeverity,
    get_error_definition,
)
from src.observability.exceptions import ClaimDenialError

logger = structlog.get_logger(__name__)


@dataclass
class ErrorEvent:
    """Single error occurrence."""

    event_id: str
    timestamp: str
    error_code: str
    component: str
    stage: str
    severity: str
    category: str
    message: str
    retryable: bool
    occurrence_key: str
    occurrence_count: int
    is_repeated: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    exception_type: str | None = None
    exception_message: str | None = None
    traceback_digest: str | None = None


class ErrorTracker:
    """
    Persist and summarize errors.

    Parameters
    ----------
    log_dir : Directory where JSONL + summary files are written.
    emit_emf: True prints CloudWatch EMF metric JSON to stdout.
    repeat_alert_threshold: count at which is_repeated becomes True.
    """

    def __init__(
        self,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        *,
        emit_emf: bool = False,
        repeat_alert_threshold: int = ERROR_REPEAT_ALERT_THRESHOLD,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / ERROR_EVENTS_FILENAME
        self.summary_path = self.log_dir / ERROR_SUMMARY_FILENAME
        self.emit_emf = emit_emf
        self.repeat_alert_threshold = repeat_alert_threshold

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        try:
            with open(self.summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Never let observability failures break the pipeline.
            logger.warning("error_summary_load_failed", path=str(self.summary_path))
            return {}

    def _write_summary(self, summary: dict[str, Any]) -> None:
        tmp = self.summary_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        tmp.replace(self.summary_path)

    def _append_event(self, event: ErrorEvent) -> None:
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

    def _traceback_digest(self, exc: BaseException | None) -> str | None:
        if exc is None:
            return None
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return hashlib.sha256(tb.encode("utf-8")).hexdigest()[:16]

    def _stable_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        return {
            key: metadata[key]
            for key in ERROR_FINGERPRINT_FIELDS
            if key in metadata and metadata[key] is not None
        }

    def _occurrence_key(
        self,
        *,
        error_code: str,
        component: str,
        stage: str,
        metadata: dict[str, Any] | None,
        exc: BaseException | None,
    ) -> str:
        payload = {
            "error_code": error_code,
            "component": component,
            "stage": stage,
            "metadata": self._stable_metadata(metadata),
            "exception_type": type(exc).__name__ if exc else None,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def record(
        self,
        code: ErrorCode | str,
        message: str | None = None,
        *,
        component: str,
        stage: str = "unknown",
        severity: ErrorSeverity | str | None = None,
        metadata: dict[str, Any] | None = None,
        exc: BaseException | None = None,
    ) -> ErrorEvent:
        """Record one error event and return the structured event."""
        try:
            error_code = code if isinstance(code, ErrorCode) else ErrorCode(code)
            definition = get_error_definition(error_code)
            effective_severity = severity or definition.default_severity
            if isinstance(effective_severity, ErrorSeverity):
                severity_value = effective_severity.value
            else:
                severity_value = str(effective_severity)

            occurrence_key = self._occurrence_key(
                error_code=error_code.value,
                component=component,
                stage=stage,
                metadata=metadata,
                exc=exc,
            )

            summary = self._load_summary()
            now = self._now()
            current = summary.get(occurrence_key, {})
            count = int(current.get("count", 0)) + 1
            summary[occurrence_key] = {
                "count": count,
                "first_seen": current.get("first_seen", now),
                "last_seen": now,
                "error_code": error_code.value,
                "component": component,
                "stage": stage,
                "severity": severity_value,
                "category": definition.category.value,
                "name": definition.name,
                "last_message": message or definition.user_message,
                "stable_metadata": self._stable_metadata(metadata),
                "retryable": definition.retryable,
            }
            self._write_summary(summary)

            event = ErrorEvent(
                event_id=str(uuid.uuid4()),
                timestamp=now,
                error_code=error_code.value,
                component=component,
                stage=stage,
                severity=severity_value,
                category=definition.category.value,
                message=message or definition.user_message,
                retryable=definition.retryable,
                occurrence_key=occurrence_key,
                occurrence_count=count,
                is_repeated=count >= self.repeat_alert_threshold,
                metadata=metadata or {},
                exception_type=type(exc).__name__ if exc else None,
                exception_message=str(exc) if exc else None,
                traceback_digest=self._traceback_digest(exc),
            )
            self._append_event(event)

            log_method = logger.warning if severity_value in {"WARNING", "INFO"} else logger.error
            log_method(
                "application_error_recorded",
                error_code=event.error_code,
                component=event.component,
                stage=event.stage,
                severity=event.severity,
                occurrence_count=event.occurrence_count,
                is_repeated=event.is_repeated,
                message=event.message,
                metadata=event.metadata,
            )

            if self.emit_emf:
                print(
                    build_error_emf_json(
                        error_code=event.error_code,
                        component=component,
                        stage=stage,
                        severity=severity_value,
                        count=count,
                    ),
                    file=sys.stdout,
                )

            return event
        except Exception as tracker_exc:
            # Observability must never break the production path.
            logger.warning("error_tracker_failed", error=str(tracker_exc))
            fallback_code = code.value if isinstance(code, ErrorCode) else str(code)
            return ErrorEvent(
                event_id=str(uuid.uuid4()),
                timestamp=self._now(),
                error_code=fallback_code,
                component=component,
                stage=stage,
                severity="WARNING",
                category="observability",
                message=f"Error tracker failed while recording: {message}",
                retryable=True,
                occurrence_key="tracker_failed",
                occurrence_count=1,
                is_repeated=False,
                metadata=metadata or {},
                exception_type=type(tracker_exc).__name__,
                exception_message=str(tracker_exc),
            )

    def record_exception(
        self,
        exc: BaseException,
        *,
        component: str,
        stage: str = "unknown",
        fallback_code: ErrorCode = ErrorCode.SYSTEM_UNEXPECTED,
        metadata: dict[str, Any] | None = None,
    ) -> ErrorEvent:
        """Record an exception, preserving ClaimDenialError codes when present."""
        if isinstance(exc, ClaimDenialError):
            merged_metadata = {**exc.metadata, **(metadata or {})}
            return self.record(
                exc.code,
                exc.message,
                component=exc.component or component,
                stage=stage,
                metadata=merged_metadata,
                exc=exc,
            )
        return self.record(
            fallback_code,
            str(exc),
            component=component,
            stage=stage,
            metadata=metadata,
            exc=exc,
        )

    def get_summary(self) -> dict[str, Any]:
        """Return the current error summary."""
        return self._load_summary()

    def get_repeated_errors(self, min_count: int = 2) -> list[dict[str, Any]]:
        """Return repeated errors sorted by count descending."""
        summary = self._load_summary()
        rows = []
        for key, value in summary.items():
            if int(value.get("count", 0)) >= min_count:
                rows.append({"occurrence_key": key, **value})
        return sorted(rows, key=lambda r: int(r.get("count", 0)), reverse=True)
