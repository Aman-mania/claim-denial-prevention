"""Schemas for custom-claim validation.

The rule layer is intentionally lightweight. It blocks structurally invalid
requests but keeps denial-risk signals such as missing diagnosis/procedure/amount
as warnings so the ML + XAI + RAG layers can still evaluate the claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

IssueSeverity = Literal["ERROR", "WARNING", "INFO"]


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding for a custom claim."""

    code: str
    message: str
    field: str | None = None
    severity: IssueSeverity = "WARNING"
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimValidationResult:
    """Result returned by the lightweight claim rule validator."""

    is_valid: bool
    normalized_claim: dict[str, Any]
    blocking_errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    infos: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "normalized_claim": self.normalized_claim,
            "blocking_errors": [issue.to_dict() for issue in self.blocking_errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "infos": [issue.to_dict() for issue in self.infos],
        }
