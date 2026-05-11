"""Local vector store for Week 6 RAG.

The local implementation prefers FAISS when available, but it is not a hard
runtime requirement. When FAISS is unavailable, the store automatically falls
back to a persisted NumPy matrix and brute-force inner-product search. This keeps
local development reliable while preserving the same VectorStore boundary for a
future Databricks Vector Search implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker
from src.rag.schemas import PolicySearchResult

logger = structlog.get_logger(__name__)

INDEX_FILENAME = "policy.faiss"
VECTOR_MATRIX_FILENAME = "policy_vectors.npy"
METADATA_FILENAME = "policy_metadata.json"
VectorBackend = Literal["auto", "faiss", "numpy"]


def _as_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    return np.ascontiguousarray(arr)


def _as_float32_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError("query vector must be shape (dim,) or (1, dim)")
    return np.ascontiguousarray(arr)


def _row_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = _as_float32_matrix(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype="float32")


class LocalFaissVectorStore:
    """Local vector store with FAISS-preferred and NumPy fallback backends.

    The class name is kept for backward compatibility with earlier Week 6 code.
    Its behavior is now backend-aware:
      - vector_backend="auto" uses FAISS if installed; otherwise NumPy search.
      - vector_backend="faiss" requires FAISS at build time.
      - vector_backend="numpy" never imports FAISS.

    Both backends persist the normalized vector matrix as ``policy_vectors.npy``.
    That means a FAISS-built index can still be searched through the NumPy
    fallback later if FAISS is unavailable in the active environment.
    """

    def __init__(
        self,
        *,
        vector_dir: Path,
        error_tracker: ErrorTracker | None = None,
        vector_backend: VectorBackend = "auto",
    ) -> None:
        self.vector_dir = Path(vector_dir)
        self.index_path = self.vector_dir / INDEX_FILENAME
        self.vector_matrix_path = self.vector_dir / VECTOR_MATRIX_FILENAME
        self.metadata_path = self.vector_dir / METADATA_FILENAME
        self.error_tracker = error_tracker or ErrorTracker()
        self.vector_backend = (vector_backend or "auto").strip().lower()  # type: ignore[assignment]
        self._index = None
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []
        self._payload: dict[str, Any] = {}
        self._loaded_backend: str | None = None

    def _try_import_faiss(self):
        try:
            import faiss
            return faiss
        except Exception:
            return None

    def _import_faiss_or_raise(self):
        faiss = self._try_import_faiss()
        if faiss is None:
            raise ClaimDenialError(
                ErrorCode.RAG_VECTOR_INDEX_MISSING,
                (
                    "FAISS is not installed in the active Python environment. "
                    "Install faiss-cpu, or use the local NumPy vector backend."
                ),
                component="rag",
                metadata={"stage": "import_faiss"},
            )
        return faiss

    def _select_build_backend(self) -> str:
        backend = self.vector_backend
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError(f"Unknown vector backend: {backend}")
        if backend == "numpy":
            return "numpy"
        if backend == "faiss":
            self._import_faiss_or_raise()
            return "faiss"
        return "faiss" if self._try_import_faiss() is not None else "numpy"

    def build(
        self,
        *,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        embedding_backend: str | None = None,
        embedding_model: str | None = None,
        embedding_metadata: dict[str, Any] | None = None,
        vector_backend: VectorBackend | None = None,
    ) -> dict[str, Any]:
        embeddings = _row_normalize(embeddings)
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata row count must match embedding row count")
        if embeddings.shape[0] == 0:
            raise ValueError("cannot build a vector index with zero embeddings")

        if vector_backend is not None:
            self.vector_backend = vector_backend
        selected_backend = self._select_build_backend()

        self.vector_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vector_matrix_path, embeddings)

        faiss_index_written = False
        if selected_backend == "faiss":
            faiss = self._import_faiss_or_raise()
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings)
            faiss.write_index(index, str(self.index_path))
            self._index = index
            faiss_index_written = True
            logger.info(
                "faiss_policy_index_built",
                rows=len(metadata),
                dim=int(embeddings.shape[1]),
                path=str(self.index_path),
            )
        else:
            self._index = None
            logger.warning(
                "faiss_unavailable_using_numpy_vector_backend",
                rows=len(metadata),
                dim=int(embeddings.shape[1]),
                vector_matrix_path=str(self.vector_matrix_path),
            )

        embedding_metadata = dict(embedding_metadata or {})
        payload = {
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_backend": embedding_backend or embedding_metadata.get("embedding_backend") or "unknown",
            "embedding_model": embedding_model or embedding_metadata.get("embedding_model") or "unknown",
            "vector_backend": selected_backend,
            "faiss_index_written": bool(faiss_index_written),
            "vector_matrix_path": str(self.vector_matrix_path),
            "index_path": str(self.index_path) if faiss_index_written else str(self.vector_matrix_path),
            "row_count": int(embeddings.shape[0]),
            "metadata": metadata,
            **{k: v for k, v in embedding_metadata.items() if k not in {"metadata"}},
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._vectors = embeddings
        self._metadata = metadata
        self._payload = payload
        self._loaded_backend = selected_backend
        logger.info(
            "policy_vector_store_built",
            rows=len(metadata),
            dim=int(embeddings.shape[1]),
            vector_backend=selected_backend,
            embedding_backend=payload["embedding_backend"],
            embedding_model=payload["embedding_model"],
            metadata_path=str(self.metadata_path),
        )
        return {
            "index_path": payload["index_path"],
            "metadata_path": str(self.metadata_path),
            "vector_matrix_path": str(self.vector_matrix_path),
            "row_count": len(metadata),
            "embedding_backend": payload["embedding_backend"],
            "embedding_model": payload["embedding_model"],
            "vector_backend": selected_backend,
            "faiss_index_written": bool(faiss_index_written),
        }

    def load(self) -> "LocalFaissVectorStore":
        if not self.metadata_path.exists():
            raise ClaimDenialError(
                ErrorCode.RAG_VECTOR_INDEX_MISSING,
                f"Vector metadata missing under {self.vector_dir}. Run run_policy_ingest.py first.",
                component="rag",
                metadata={"stage": "load_vector_index", "path": str(self.vector_dir)},
            )

        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._payload = payload
        self._metadata = list(payload.get("metadata", []))
        requested_backend = self.vector_backend
        stored_backend = str(payload.get("vector_backend") or "auto")

        matrix_path = Path(payload.get("vector_matrix_path") or self.vector_matrix_path)
        if not matrix_path.is_absolute():
            matrix_path = self.vector_dir / matrix_path
        if matrix_path.exists():
            self._vectors = _row_normalize(np.load(matrix_path))
        else:
            self._vectors = None

        use_faiss = False
        if requested_backend == "faiss":
            use_faiss = True
        elif requested_backend == "numpy":
            use_faiss = False
        else:
            use_faiss = stored_backend == "faiss" and self.index_path.exists() and self._try_import_faiss() is not None

        if use_faiss:
            faiss = self._import_faiss_or_raise()
            if not self.index_path.exists():
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    f"FAISS index missing: {self.index_path}",
                    component="rag",
                    metadata={"stage": "load_vector_index", "path": str(self.index_path)},
                )
            self._index = faiss.read_index(str(self.index_path))
            self._loaded_backend = "faiss"
        else:
            if self._vectors is None:
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    (
                        f"Vector matrix missing: {matrix_path}. Cannot use NumPy fallback. "
                        "Re-run run_policy_ingest.py."
                    ),
                    component="rag",
                    metadata={"stage": "load_vector_index", "path": str(matrix_path)},
                )
            self._index = None
            self._loaded_backend = "numpy"

        logger.info(
            "policy_vector_store_loaded",
            rows=len(self._metadata),
            stored_vector_backend=stored_backend,
            loaded_vector_backend=self._loaded_backend,
            embedding_backend=payload.get("embedding_backend"),
            embedding_model=payload.get("embedding_model"),
        )
        return self

    @property
    def metadata_payload(self) -> dict[str, Any]:
        if not self._payload and self.metadata_path.exists():
            self._payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._payload

    def _search_numpy(self, query: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            if self.vector_matrix_path.exists():
                self._vectors = _row_normalize(np.load(self.vector_matrix_path))
            else:
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    "Vector matrix not loaded. Run run_policy_ingest.py first.",
                    component="rag",
                    metadata={"stage": "search_numpy"},
                )
        scores = (self._vectors @ query.reshape(-1)).astype("float32")
        if scores.size == 0:
            return np.asarray([], dtype="float32"), np.asarray([], dtype="int64")
        k = max(1, min(int(top_k), scores.size))
        # Stable sort by descending score; mergesort preserves corpus order for ties.
        indices = np.argsort(-scores, kind="mergesort")[:k]
        return scores[indices], indices.astype("int64")

    def search(self, query_vector: np.ndarray, *, top_k: int = 5, min_score: float | None = None) -> list[PolicySearchResult]:
        if self._loaded_backend is None:
            self.load()
        query = _row_normalize(_as_float32_vector(query_vector))
        k = max(1, min(int(top_k), len(self._metadata)))

        if self._loaded_backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(query, k)
            score_list = scores[0].tolist()
            index_list = indices[0].tolist()
        else:
            scores, indices = self._search_numpy(query, top_k=k)
            score_list = scores.tolist()
            index_list = indices.tolist()

        results: list[PolicySearchResult] = []
        for score, idx in zip(score_list, index_list):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if min_score is not None and float(score) < float(min_score):
                continue
            meta = self._metadata[idx]
            results.append(
                PolicySearchResult(
                    chunk_id=str(meta.get("chunk_id")),
                    score=float(score),
                    raw_score=float(score),
                    tag_overlap_count=0,
                    metadata=meta,
                )
            )
        return results


# Clearer alias for new code. Kept alongside LocalFaissVectorStore for backward compatibility.
LocalVectorStore = LocalFaissVectorStore
