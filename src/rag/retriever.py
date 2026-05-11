"""Reason-aware policy retrieval for Week 6 RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker
from src.rag.embedder import SentenceTransformerEmbedder, embedder_from_vector_metadata
from src.rag.schemas import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, PolicySearchResult
from src.rag.vector_store import LocalVectorStore

logger = structlog.get_logger(__name__)


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, tuple):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(x) for x in decoded]
        except Exception:
            return [x.strip() for x in raw.split(",") if x.strip()]
    return []


class PolicyRetriever:
    """Retrieve policy chunks using reason query + optional policy tags."""

    def __init__(
        self,
        *,
        vector_store: LocalVectorStore,
        embedder: SentenceTransformerEmbedder,
        error_tracker: ErrorTracker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.error_tracker = error_tracker or ErrorTracker()

    @classmethod
    def load(
        cls,
        *,
        vector_dir: Path,
        embedder: SentenceTransformerEmbedder | None = None,
        error_tracker: ErrorTracker | None = None,
    ) -> "PolicyRetriever":
        tracker = error_tracker or ErrorTracker()
        store = LocalVectorStore(vector_dir=vector_dir, error_tracker=tracker).load()
        return cls(
            vector_store=store,
            embedder=embedder or embedder_from_vector_metadata(vector_dir=vector_dir, error_tracker=tracker),
            error_tracker=tracker,
        )

    def retrieve(
        self,
        *,
        query: str,
        policy_tags: list[str] | tuple[str, ...] | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[PolicySearchResult]:
        """Retrieve relevant policies for one reason-aware query.

        We over-fetch slightly, then boost results that share reason tags. This
        keeps the local implementation simple and makes the behavior close to a
        future vector search + metadata filter/reranker setup.
        """
        try:
            tags = [str(tag) for tag in (policy_tags or [])]
            query_text = str(query or "").strip()
            if not query_text:
                raise ValueError("retrieval query cannot be empty")

            query_vector = self.embedder.embed_query(query_text)
            raw_results = self.vector_store.search(query_vector, top_k=max(top_k * 4, top_k), min_score=None)

            scored: list[PolicySearchResult] = []
            query_tag_set = set(tags)
            for result in raw_results:
                chunk_tags = set(_parse_tags(result.metadata.get("policy_tags_json")))
                overlap = len(query_tag_set & chunk_tags)
                boosted = float(result.raw_score) + (0.04 * overlap)
                if boosted < min_score:
                    continue
                scored.append(
                    PolicySearchResult(
                        chunk_id=result.chunk_id,
                        score=boosted,
                        raw_score=result.raw_score,
                        tag_overlap_count=overlap,
                        metadata=result.metadata,
                    )
                )

            scored.sort(key=lambda item: (item.score, item.tag_overlap_count), reverse=True)
            selected = scored[:top_k]
            if not selected:
                self.error_tracker.record(
                    ErrorCode.RAG_NO_RELEVANT_POLICY_FOUND,
                    "No relevant policy met the retrieval threshold.",
                    component="rag",
                    stage="retrieve_policy",
                    metadata={"stage": "retrieve_policy", "top_k": top_k, "min_score": min_score},
                )
            return selected
        except ClaimDenialError:
            raise
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="retrieve_policy",
                fallback_code=ErrorCode.RAG_RETRIEVAL_FAILED,
                metadata={"stage": "retrieve_policy"},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_RETRIEVAL_FAILED,
                f"Policy retrieval failed: {exc}",
                component="rag",
                metadata={"stage": "retrieve_policy"},
            ) from exc
