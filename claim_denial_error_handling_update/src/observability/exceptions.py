"""Application exceptions carrying structured error codes."""

from __future__ import annotations

from typing import Any

from src.observability.error_codes import ErrorCode, get_error_definition


class ClaimDenialError(Exception):
    """
    Base exception for known project errors.

    Raise this when the error is expected/operational and should have a stable
    error code. Unknown exceptions are still caught and wrapped by the tracker.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        definition = get_error_definition(code)
        self.message = message or definition.user_message
        self.component = component
        self.metadata = metadata or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        definition = get_error_definition(self.code)
        return {
            "error_code": self.code.value,
            "name": definition.name,
            "message": self.message,
            "category": definition.category.value,
            "severity": definition.default_severity.value,
            "retryable": definition.retryable,
            "component": self.component,
            "metadata": self.metadata,
        }
