"""Serializable schemas for the Week 7 remediation agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Priority = Literal["LOW", "MEDIUM", "HIGH"]
DecisionStatus = Literal["READY_TO_SUBMIT", "REVIEW_RECOMMENDED", "REVIEW_REQUIRED", "BLOCKED"]


@dataclass(frozen=True)
class PolicyEvidence:
    source_name: str | None
    section_title: str | None
    policy_summary: str
    similarity_score: float | None = None
    policy_chunk_id: str | None = None
    reason_code: str | None = None
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    action: str
    priority: Priority
    reason_code: str | None = None
    why: str | None = None
    evidence_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDecision:
    status: DecisionStatus
    priority: Priority
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentPresentation:
    source: str
    summary: str
    action_plan: list[str] = field(default_factory=list)
    analyst_notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
