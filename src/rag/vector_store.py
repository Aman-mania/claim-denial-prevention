"""Local vector store for policy RAG.

The Docker/AWS default is now a scikit-learn nearest-neighbor index. This avoids
FAISS and avoids the previous NumPy-only search backend while keeping artifacts
portable. FAISS remains available as an optional local backend. The old NumPy
backend is kept only for backward compatibility with existing artifacts and is
not used by the Docker/AWS default.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker
from src.rag.schemas import PolicySearchResult

logger = structlog.get_logger(__name__)

INDEX_FILENAME = "policy.faiss"
SKLEARN_INDEX_FILENAME = "policy_sklearn_nn.pkl"
VECTOR_MATRIX_FILENAME = "policy_vectors.npy"
METADATA_FILENAME = "policy_metadata.json"
VectorBackend = Literal["auto", "sklearn", "faiss", "numpy"]


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
    """Local vector store with sklearn default and optional FAISS backend.

    Class name is retained for backward compatibility. New code should use the
    alias ``LocalVectorStore`` below.
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
        self.sklearn_index_path = self.vector_dir / SKLEARN_INDEX_FILENAME
        self.vector_matrix_path = self.vector_dir / VECTOR_MATRIX_FILENAME
        self.metadata_path = self.vector_dir / METADATA_FILENAME
        self.error_tracker = error_tracker or ErrorTracker()
        self.vector_backend = (vector_backend or "auto").strip().lower()  # type: ignore[assignment]
        self._index = None
        self._sklearn_index = None
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
                "FAISS is not installed. Use vector_backend=sklearn for Docker/AWS.",
                component="rag",
                metadata={"stage": "import_faiss"},
            )
        return faiss

    def _new_sklearn_index(self, embeddings: np.ndarray):
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as exc:
            raise ClaimDenialError(
                ErrorCode.RAG_VECTOR_INDEX_MISSING,
                "scikit-learn NearestNeighbors is required for vector_backend=sklearn.",
                component="rag",
                metadata={"stage": "load_sklearn_nearest_neighbors"},
            ) from exc
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        return nn

    def _select_build_backend(self) -> str:
        backend = self.vector_backend
        if backend not in {"auto", "sklearn", "faiss", "numpy"}:
            raise ValueError(f"Unknown vector backend: {backend}")
        if backend == "auto":
            return "sklearn"
        if backend == "faiss":
            self._import_faiss_or_raise()
        return backend

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
        sklearn_index_written = False
        if selected_backend == "faiss":
            faiss = self._import_faiss_or_raise()
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings)
            faiss.write_index(index, str(self.index_path))
            self._index = index
            faiss_index_written = True
            index_path = str(self.index_path)
            logger.info("faiss_policy_index_built", rows=len(metadata), dim=int(embeddings.shape[1]), path=index_path)
        elif selected_backend == "sklearn":
            nn = self._new_sklearn_index(embeddings)
            with open(self.sklearn_index_path, "wb") as f:
                pickle.dump(nn, f)
            self._sklearn_index = nn
            sklearn_index_written = True
            index_path = str(self.sklearn_index_path)
            logger.info("sklearn_policy_index_built", rows=len(metadata), dim=int(embeddings.shape[1]), path=index_path)
        else:
            # Backward compatibility only. Do not use for Docker/AWS deployment.
            self._index = None
            index_path = str(self.vector_matrix_path)
            logger.warning("numpy_vector_backend_selected_backward_compatibility", rows=len(metadata), dim=int(embeddings.shape[1]))

        embedding_metadata = dict(embedding_metadata or {})
        payload = {
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_backend": embedding_backend or embedding_metadata.get("embedding_backend") or "unknown",
            "embedding_model": embedding_model or embedding_metadata.get("embedding_model") or "unknown",
            "vector_backend": selected_backend,
            "faiss_index_written": bool(faiss_index_written),
            "sklearn_index_written": bool(sklearn_index_written),
            "vector_matrix_path": str(self.vector_matrix_path),
            "sklearn_index_path": str(self.sklearn_index_path) if sklearn_index_written else None,
            "index_path": index_path,
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
            "sklearn_index_written": bool(sklearn_index_written),
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
        stored_backend = str(payload.get("vector_backend") or "sklearn")

        matrix_path = Path(payload.get("vector_matrix_path") or self.vector_matrix_path)
        if not matrix_path.is_absolute():
            matrix_path = self.vector_dir / matrix_path
        self._vectors = _row_normalize(np.load(matrix_path)) if matrix_path.exists() else None

        selected_backend = requested_backend if requested_backend != "auto" else stored_backend
        if selected_backend == "faiss":
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
        elif selected_backend == "sklearn":
            sklearn_path = Path(payload.get("sklearn_index_path") or self.sklearn_index_path)
            if not sklearn_path.is_absolute():
                sklearn_path = self.vector_dir / sklearn_path
            if not sklearn_path.exists():
                if self._vectors is None:
                    raise ClaimDenialError(
                        ErrorCode.RAG_VECTOR_INDEX_MISSING,
                        f"sklearn index and vector matrix are both missing under {self.vector_dir}.",
                        component="rag",
                        metadata={"stage": "load_sklearn_vector_index"},
                    )
                # Rebuild in-memory index if matrix exists but pickle is absent.
                self._sklearn_index = self._new_sklearn_index(self._vectors)
            else:
                with open(sklearn_path, "rb") as f:
                    self._sklearn_index = pickle.load(f)
            self._loaded_backend = "sklearn"
        else:
            if self._vectors is None:
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    f"Vector matrix missing: {matrix_path}.",
                    component="rag",
                    metadata={"stage": "load_numpy_compat_vector_index", "path": str(matrix_path)},
                )
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

    def _search_numpy_compat(self, query: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            if self.vector_matrix_path.exists():
                self._vectors = _row_normalize(np.load(self.vector_matrix_path))
            else:
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    "Vector matrix not loaded. Run run_policy_ingest.py first.",
                    component="rag",
                    metadata={"stage": "search_numpy_compat"},
                )
        scores = (self._vectors @ query.reshape(-1)).astype("float32")
        if scores.size == 0:
            return np.asarray([], dtype="float32"), np.asarray([], dtype="int64")
        k = max(1, min(int(top_k), scores.size))
        indices = np.argsort(-scores, kind="mergesort")[:k]
        return scores[indices], indices.astype("int64")

    def _search_sklearn(self, query: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._sklearn_index is None:
            if self._vectors is None:
                raise ClaimDenialError(
                    ErrorCode.RAG_VECTOR_INDEX_MISSING,
                    "sklearn index not loaded. Run run_policy_ingest.py first.",
                    component="rag",
                    metadata={"stage": "search_sklearn"},
                )
            self._sklearn_index = self._new_sklearn_index(self._vectors)
        distances, indices = self._sklearn_index.kneighbors(query, n_neighbors=top_k)
        # NearestNeighbors with cosine returns distance where 0 is identical.
        scores = (1.0 - distances[0]).astype("float32")
        return scores, indices[0].astype("int64")

    def search(self, query_vector: np.ndarray, *, top_k: int = 5, min_score: float | None = None) -> list[PolicySearchResult]:
        if self._loaded_backend is None:
            self.load()
        query = _row_normalize(_as_float32_vector(query_vector))
        k = max(1, min(int(top_k), len(self._metadata)))

        if self._loaded_backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(query, k)
            score_list = scores[0].tolist()
            index_list = indices[0].tolist()
        elif self._loaded_backend == "sklearn":
            scores, indices = self._search_sklearn(query, top_k=k)
            score_list = scores.tolist()
            index_list = indices.tolist()
        else:
            scores, indices = self._search_numpy_compat(query, top_k=k)
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


LocalVectorStore = LocalFaissVectorStore
