"""Convenience wrappers for pipeline entry points."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from src.observability.error_codes import ErrorCode
from src.observability.error_tracker import ErrorTracker

F = TypeVar("F", bound=Callable[..., Any])


def tracked_errors(
    *,
    component: str,
    stage: str,
    fallback_code: ErrorCode = ErrorCode.SYSTEM_UNEXPECTED,
    tracker: ErrorTracker | None = None,
) -> Callable[[F], F]:
    """Decorator that records exceptions then re-raises them."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            local_tracker = tracker or ErrorTracker()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                local_tracker.record_exception(
                    exc,
                    component=component,
                    stage=stage,
                    fallback_code=fallback_code,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
