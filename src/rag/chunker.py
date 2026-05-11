"""Policy chunking for Week 6 RAG.

Chunks are reason-retrieval units. They intentionally preserve source metadata
and inferred policy tags so retrieval can combine vector similarity with simple
metadata matching.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import structlog

from src.rag.schemas import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_MODEL,
    POLICY_CHUNK_COLUMNS,
    RAG_VERSION,
    PolicyChunk,
    PolicyDocument,
)

logger = structlog.get_logger(__name__)

_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "diagnosis": ("diagnosis", "diagnostic", "icd", "icd-10"),
    "medical_necessity": ("medical necessity", "medically necessary", "clinical support", "clinical justification"),
    "claim_completeness": ("required field", "complete claim", "incomplete", "missing", "must include"),
    "procedure_coding": ("procedure", "cpt", "hcpcs", "code combination", "coding edit", "coding"),
    "documentation": ("documentation", "supporting document", "records", "attachment", "evidence"),
    "prior_authorization": ("prior authorization", "pre-authorization", "preauthorization", "pre-claim"),
    "high_cost": ("high cost", "above expected", "cost benchmark", "unusual charge", "excessive"),
    "provider": ("provider", "billing provider", "rendering provider", "credential"),
    "duplicate": ("duplicate", "resubmission", "same service"),
    "payer_policy": ("payer", "insurer", "coverage", "payment policy"),
}

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(?P<title>.+?)\s*$")


def infer_policy_tags(text: str) -> tuple[str, ...]:
    """Infer coarse policy tags using deterministic keyword matching."""
    lowered = text.lower()
    tags: list[str] = []
    for tag, keywords in _TAG_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return tuple(tags)


def _chunk_id(document_id: str, section_title: str | None, page_number: int | None, chunk_index: int, text: str) -> str:
    payload = f"{document_id}|{section_title}|{page_number}|{chunk_index}|{text[:160]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown-ish text by headings while supporting plain text."""
    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = match.group("title").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    if not sections:
        return [(None, text)]
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def _word_windows(words: list[str], *, chunk_size: int, overlap: int) -> Iterable[tuple[int, list[str]]]:
    if not words:
        return
    step = max(1, chunk_size - overlap)
    start = 0
    idx = 0
    while start < len(words):
        yield idx, words[start:start + chunk_size]
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step


class PolicyChunker:
    """Split policy documents into metadata-rich chunks."""

    def __init__(
        self,
        *,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be >= 0 and smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.embedding_model = embedding_model

    def chunk_documents(self, documents: list[PolicyDocument]) -> list[PolicyChunk]:
        created_at = datetime.now(timezone.utc).isoformat()
        chunks: list[PolicyChunk] = []

        for doc in documents:
            for section_title, section_text in _split_sections(doc.text):
                words = section_text.split()
                if not words:
                    continue
                for local_idx, window in _word_windows(
                    words,
                    chunk_size=self.chunk_size_words,
                    overlap=self.chunk_overlap_words,
                ):
                    chunk_text = " ".join(window).strip()
                    tags = infer_policy_tags(" ".join([section_title or "", chunk_text]))
                    global_idx = len(chunks)
                    chunks.append(
                        PolicyChunk(
                            chunk_id=_chunk_id(doc.document_id, section_title, doc.page_number, global_idx, chunk_text),
                            document_id=doc.document_id,
                            source_name=doc.source_name,
                            source_type=doc.source_type,
                            source_path=doc.source_path,
                            section_title=section_title,
                            page_number=doc.page_number,
                            chunk_index=local_idx,
                            chunk_text=chunk_text,
                            policy_tags=tags,
                            token_estimate=max(1, int(len(window) * 1.33)),
                            embedding_model=self.embedding_model,
                            rag_version=RAG_VERSION,
                            created_at=created_at,
                        )
                    )

        logger.info("policy_documents_chunked", documents=len(documents), chunks=len(chunks))
        return chunks

    def to_dataframe(self, chunks: list[PolicyChunk]) -> pd.DataFrame:
        rows = [chunk.to_row() for chunk in chunks]
        df = pd.DataFrame(rows, columns=POLICY_CHUNK_COLUMNS)
        return df
