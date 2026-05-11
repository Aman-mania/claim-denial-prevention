"""Stable schemas and table names for Week 6 policy RAG.

The RAG layer consumes Week 5 explanation rows and produces two downstream
artifacts:
1. reason-level policy matches;
2. claim-level final explanations that combine ML risk + reasons + policy.

These names are kept centralized so local Parquet tables can be swapped for
Delta/Unity Catalog tables later without changing business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RAG_VERSION = "week6_rag_v1"

# Table names. LocalTableStore writes these as <name>.parquet.
POLICY_CHUNK_TABLE = "policy_chunks"
POLICY_MATCH_TABLE = "gold_claim_policy_matches"
FINAL_EXPLANATION_TABLE = "gold_claim_final_explanations"
POLICY_INGEST_REPORT_FILE = "policy_ingest_report.json"
POLICY_MATCH_REPORT_FILE = "policy_match_report.json"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE_WORDS = 420
DEFAULT_CHUNK_OVERLAP_WORDS = 70
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.25

POLICY_CHUNK_COLUMNS: list[str] = [
    "chunk_id",
    "document_id",
    "source_name",
    "source_type",
    "source_path",
    "section_title",
    "page_number",
    "chunk_index",
    "chunk_text",
    "policy_tags_json",
    "token_estimate",
    "embedding_model",
    "rag_version",
    "created_at",
]

POLICY_MATCH_COLUMNS: list[str] = [
    "claim_id",
    "risk_score",
    "risk_level",
    "predicted_denial",
    "reason_rank",
    "reason_code",
    "reason_title",
    "reason_text",
    "fix_suggestion",
    "policy_rank",
    "policy_chunk_id",
    "policy_text",
    "policy_summary",
    "source_name",
    "source_type",
    "source_path",
    "section_title",
    "page_number",
    "similarity_score",
    "raw_similarity_score",
    "tag_overlap_count",
    "retrieval_query",
    "query_policy_tags_json",
    "retrieved_policy_tags_json",
    "rag_version",
    "created_at",
]

FINAL_EXPLANATION_COLUMNS: list[str] = [
    "claim_id",
    "risk_score",
    "risk_level",
    "predicted_denial",
    "final_explanation_text",
    "reasons_json",
    "policies_json",
    "recommended_actions_json",
    "rag_version",
    "created_at",
]


@dataclass(frozen=True)
class PolicyDocument:
    """One loadable text unit from a policy source.

    For PDFs, each page is represented as a separate PolicyDocument with the same
    document_id and a populated page_number. For TXT/MD, page_number is None.
    """

    document_id: str
    source_name: str
    source_type: str
    source_path: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class PolicyChunk:
    """A policy text chunk with retrieval metadata."""

    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    source_path: str
    section_title: str | None
    page_number: int | None
    chunk_index: int
    chunk_text: str
    policy_tags: tuple[str, ...] = field(default_factory=tuple)
    token_estimate: int = 0
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rag_version: str = RAG_VERSION
    created_at: str | None = None

    def to_row(self) -> dict[str, Any]:
        import json

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "policy_tags_json": json.dumps(list(self.policy_tags)),
            "token_estimate": self.token_estimate,
            "embedding_model": self.embedding_model,
            "rag_version": self.rag_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PolicySearchResult:
    """One retrieved policy chunk for a reason query."""

    chunk_id: str
    score: float
    raw_score: float
    tag_overlap_count: int
    metadata: dict[str, Any]
