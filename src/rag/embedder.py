"""Embedding generation for Week 6 policy RAG.

Backends
--------
1. sentence-transformers: preferred semantic local backend.
2. tfidf: strong offline/local fallback for a small policy corpus.
3. sklearn-hashing: stateless emergency fallback with no persisted vocabulary.

The design keeps ingestion/query embedding compatible by persisting any fitted
retrieval artifact (TF-IDF vectorizer) beside the vector index metadata.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker
from src.rag.schemas import DEFAULT_EMBEDDING_MODEL

logger = structlog.get_logger(__name__)

DEFAULT_HASHING_FEATURES = int(os.getenv("RAG_HASHING_FEATURES", "4096"))
DEFAULT_TFIDF_MAX_FEATURES = int(os.getenv("RAG_TFIDF_MAX_FEATURES", "8192"))
VECTOR_METADATA_FILENAME = "policy_metadata.json"
TFIDF_VECTORIZER_FILENAME = "policy_tfidf_vectorizer.pkl"
EmbeddingBackend = Literal["auto", "sentence-transformers", "tfidf", "sklearn-hashing"]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize dense vectors row-wise for cosine/IP search."""
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _safe_texts(texts: Iterable[str]) -> list[str]:
    return [str(text or "") for text in texts]


class TfidfTextEmbedder:
    """Offline retrieval embedder using a persisted TF-IDF vocabulary.

    TF-IDF is more suitable than HashingVectorizer for our small local policy
    corpus because it learns corpus-specific vocabulary and IDF weights. During
    ingestion it fits on policy chunks and saves the vectorizer. During retrieval
    it loads the same vectorizer so query vectors are compatible with the index.
    """

    backend_name = "tfidf"

    def __init__(
        self,
        *,
        artifact_dir: Path | None = None,
        vectorizer_path: Path | None = None,
        max_features: int = DEFAULT_TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = (1, 2),
        normalize_embeddings: bool = True,
        error_tracker: ErrorTracker | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path is not None else (
            self.artifact_dir / TFIDF_VECTORIZER_FILENAME if self.artifact_dir is not None else None
        )
        self.max_features = int(max_features)
        self.ngram_range = ngram_range
        self.normalize_embeddings = normalize_embeddings
        self.error_tracker = error_tracker or ErrorTracker()
        self.model_name = f"sklearn-tfidf-{self.max_features}"
        self._vectorizer = None

    def _new_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words="english",
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            norm="l2" if self.normalize_embeddings else None,
            dtype=np.float32,
        )

    def _load_vectorizer(self):
        if self._vectorizer is not None:
            return self._vectorizer
        if self.vectorizer_path and self.vectorizer_path.exists():
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self._vectorizer = pickle.load(f)
                logger.info("tfidf_vectorizer_loaded", path=str(self.vectorizer_path))
                return self._vectorizer
            except Exception as exc:
                self.error_tracker.record_exception(
                    exc,
                    component="rag",
                    stage="load_tfidf_vectorizer",
                    fallback_code=ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                    metadata={"stage": "load_tfidf_vectorizer", "path": str(self.vectorizer_path)},
                )
                raise ClaimDenialError(
                    ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                    f"Could not load persisted TF-IDF vectorizer: {self.vectorizer_path}",
                    component="rag",
                    metadata={"stage": "load_tfidf_vectorizer", "path": str(self.vectorizer_path)},
                ) from exc
        return None

    def _save_vectorizer(self) -> None:
        if self.vectorizer_path is None or self._vectorizer is None:
            return
        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self._vectorizer, f)
        logger.info("tfidf_vectorizer_saved", path=str(self.vectorizer_path))

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = _safe_texts(texts)
        if not texts:
            return np.zeros((0, self.max_features), dtype="float32")
        try:
            vectorizer = self._load_vectorizer()
            if vectorizer is None:
                vectorizer = self._new_vectorizer()
                sparse = vectorizer.fit_transform(texts)
                self._vectorizer = vectorizer
                self._save_vectorizer()
                logger.info("tfidf_embedding_backend_fitted", rows=len(texts), max_features=self.max_features)
            else:
                sparse = vectorizer.transform(texts)
            vectors = sparse.astype(np.float32).toarray()
            if self.normalize_embeddings:
                vectors = _normalize_rows(vectors)
            logger.info("tfidf_embeddings_generated", rows=len(texts), dim=int(vectors.shape[1]))
            return vectors.astype("float32", copy=False)
        except ClaimDenialError:
            raise
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="generate_tfidf_embeddings",
                fallback_code=ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                metadata={"stage": "generate_tfidf_embeddings", "count": len(texts)},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "TF-IDF embedding generation failed for policy text.",
                component="rag",
                metadata={"stage": "generate_tfidf_embeddings", "count": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> np.ndarray:
        vectorizer = self._load_vectorizer()
        if vectorizer is None:
            raise ClaimDenialError(
                ErrorCode.RAG_VECTOR_INDEX_MISSING,
                "TF-IDF vectorizer is missing. Re-run run_policy_ingest.py with --embedding-backend tfidf.",
                component="rag",
                metadata={"stage": "embed_tfidf_query", "path": str(self.vectorizer_path)},
            )
        vectors = self.embed_texts([query])
        if vectors.size == 0:
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "Query embedding was empty.",
                component="rag",
                metadata={"stage": "embed_query"},
            )
        return vectors[0]

    def metadata(self) -> dict:
        return {
            "embedding_backend": self.backend_name,
            "embedding_model": self.model_name,
            "embedding_dim": None,
            "tfidf_max_features": self.max_features,
            "tfidf_vectorizer_path": str(self.vectorizer_path) if self.vectorizer_path else None,
            "normalize_embeddings": self.normalize_embeddings,
        }


class HashingTextEmbedder:
    """Dependency-light emergency fallback using scikit-learn HashingVectorizer."""

    backend_name = "sklearn-hashing"

    def __init__(
        self,
        *,
        n_features: int = DEFAULT_HASHING_FEATURES,
        ngram_range: tuple[int, int] = (1, 2),
        normalize_embeddings: bool = True,
        error_tracker: ErrorTracker | None = None,
    ) -> None:
        self.n_features = int(n_features)
        self.ngram_range = ngram_range
        self.normalize_embeddings = normalize_embeddings
        self.error_tracker = error_tracker or ErrorTracker()
        self.model_name = f"sklearn-hashing-{self.n_features}"
        self._vectorizer = None

    def _load_vectorizer(self):
        if self._vectorizer is not None:
            return self._vectorizer
        try:
            from sklearn.feature_extraction.text import HashingVectorizer

            self._vectorizer = HashingVectorizer(
                n_features=self.n_features,
                alternate_sign=False,
                norm="l2" if self.normalize_embeddings else None,
                lowercase=True,
                stop_words="english",
                ngram_range=self.ngram_range,
                dtype=np.float32,
            )
            logger.info("hashing_embedding_backend_loaded", n_features=self.n_features)
            return self._vectorizer
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="load_hashing_embedder",
                fallback_code=ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                metadata={"stage": "load_hashing_embedder", "n_features": self.n_features},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                "Could not load sklearn HashingVectorizer fallback embedder.",
                component="rag",
                metadata={"stage": "load_hashing_embedder", "n_features": self.n_features},
            ) from exc

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = _safe_texts(texts)
        if not texts:
            return np.zeros((0, self.n_features), dtype="float32")
        try:
            vectorizer = self._load_vectorizer()
            sparse = vectorizer.transform(texts)
            vectors = sparse.astype(np.float32).toarray()
            if self.normalize_embeddings:
                vectors = _normalize_rows(vectors)
            logger.info("hashing_embeddings_generated", rows=len(texts), dim=int(vectors.shape[1]))
            return vectors.astype("float32", copy=False)
        except ClaimDenialError:
            raise
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="generate_hashing_embeddings",
                fallback_code=ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                metadata={"stage": "generate_hashing_embeddings", "count": len(texts)},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "Hashing embedding generation failed for policy text.",
                component="rag",
                metadata={"stage": "generate_hashing_embeddings", "count": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embed_texts([query])
        if vectors.size == 0:
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "Query embedding was empty.",
                component="rag",
                metadata={"stage": "embed_query"},
            )
        return vectors[0]

    def metadata(self) -> dict:
        return {
            "embedding_backend": self.backend_name,
            "embedding_model": self.model_name,
            "embedding_dim": self.n_features,
            "hashing_features": self.n_features,
            "normalize_embeddings": self.normalize_embeddings,
        }


class SentenceTransformerEmbedder:
    """Wrapper around Sentence Transformers with optional fallback."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        allow_fallback: bool | None = None,
        fallback_backend: Literal["tfidf", "sklearn-hashing"] = "tfidf",
        artifact_dir: Path | None = None,
        hashing_features: int = DEFAULT_HASHING_FEATURES,
        tfidf_max_features: int = DEFAULT_TFIDF_MAX_FEATURES,
        error_tracker: ErrorTracker | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.allow_fallback = (
            os.getenv("RAG_ALLOW_EMBEDDING_FALLBACK", "true").strip().lower() not in {"0", "false", "no"}
            if allow_fallback is None else bool(allow_fallback)
        )
        self.fallback_backend = fallback_backend
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.hashing_features = int(hashing_features)
        self.tfidf_max_features = int(tfidf_max_features)
        self.error_tracker = error_tracker or ErrorTracker()
        self._model = None
        self._fallback: TfidfTextEmbedder | HashingTextEmbedder | None = None
        self.backend_name = "sentence-transformers"
        self.effective_model_name = self.model_name

    def _fallback_embedder(self, reason: Exception | None = None):
        if not self.allow_fallback:
            assert reason is not None
            raise reason
        if self._fallback is None:
            logger.warning(
                "sentence_transformer_unavailable_using_embedding_fallback",
                model_name=self.model_name,
                fallback_backend=self.fallback_backend,
                reason=str(reason) if reason else None,
            )
            if self.fallback_backend == "sklearn-hashing":
                self._fallback = HashingTextEmbedder(
                    n_features=self.hashing_features,
                    normalize_embeddings=self.normalize_embeddings,
                    error_tracker=self.error_tracker,
                )
            else:
                self._fallback = TfidfTextEmbedder(
                    artifact_dir=self.artifact_dir,
                    max_features=self.tfidf_max_features,
                    normalize_embeddings=self.normalize_embeddings,
                    error_tracker=self.error_tracker,
                )
            self.backend_name = self._fallback.backend_name
            self.effective_model_name = self._fallback.model_name
        return self._fallback

    def _load_model(self):
        if self._fallback is not None:
            return self._fallback
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.backend_name = "sentence-transformers"
            self.effective_model_name = self.model_name
            logger.info("embedding_model_loaded", model_name=self.model_name)
            return self._model
        except ModuleNotFoundError as exc:
            if self.allow_fallback:
                return self._fallback_embedder(exc)
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="load_embedding_model",
                fallback_code=ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                metadata={"stage": "load_embedding_model", "model_name": self.model_name},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                "sentence-transformers is not installed. Install it or use --embedding-backend tfidf.",
                component="rag",
                metadata={"stage": "load_embedding_model", "model_name": self.model_name},
            ) from exc
        except Exception as exc:
            if self.allow_fallback:
                return self._fallback_embedder(exc)
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="load_embedding_model",
                fallback_code=ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                metadata={"stage": "load_embedding_model", "model_name": self.model_name},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED,
                f"Could not load embedding model: {self.model_name}",
                component="rag",
                metadata={"stage": "load_embedding_model", "model_name": self.model_name},
            ) from exc

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = _safe_texts(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        try:
            model = self._load_model()
            if isinstance(model, (HashingTextEmbedder, TfidfTextEmbedder)):
                return model.embed_texts(texts)
            try:
                vectors = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            except TypeError:
                vectors = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                if self.normalize_embeddings:
                    vectors = _normalize_rows(np.asarray(vectors, dtype="float32"))
            vectors = np.asarray(vectors, dtype="float32")
            if self.normalize_embeddings:
                vectors = _normalize_rows(vectors)
            self.backend_name = "sentence-transformers"
            self.effective_model_name = self.model_name
            logger.info("embeddings_generated", rows=len(texts), dim=int(vectors.shape[1]))
            return vectors.astype("float32", copy=False)
        except ClaimDenialError:
            raise
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="generate_embeddings",
                fallback_code=ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                metadata={"stage": "generate_embeddings", "count": len(texts)},
            )
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "Embedding generation failed for policy text.",
                component="rag",
                metadata={"stage": "generate_embeddings", "count": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embed_texts([query])
        if vectors.size == 0:
            raise ClaimDenialError(
                ErrorCode.RAG_EMBEDDING_GENERATION_FAILED,
                "Query embedding was empty.",
                component="rag",
                metadata={"stage": "embed_query"},
            )
        return vectors[0]

    def metadata(self) -> dict:
        if self._fallback is not None:
            return self._fallback.metadata()
        return {
            "embedding_backend": self.backend_name,
            "embedding_model": self.effective_model_name,
            "normalize_embeddings": self.normalize_embeddings,
        }


def create_embedder(
    *,
    backend: EmbeddingBackend = "auto",
    model_name: str | None = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    allow_fallback: bool = True,
    fallback_backend: Literal["tfidf", "sklearn-hashing"] = "tfidf",
    artifact_dir: Path | None = None,
    hashing_features: int = DEFAULT_HASHING_FEATURES,
    tfidf_max_features: int = DEFAULT_TFIDF_MAX_FEATURES,
    error_tracker: ErrorTracker | None = None,
):
    """Factory for local embedding backends.

    `auto` prefers Sentence Transformers and falls back to TF-IDF by default.
    `tfidf` is the recommended offline fallback for the small policy corpus.
    `sklearn-hashing` remains available as a stateless emergency fallback.
    """
    backend = (backend or "auto").strip().lower()  # type: ignore[assignment]
    if backend in {"tfidf", "sklearn-tfidf", "local-tfidf"}:
        return TfidfTextEmbedder(
            artifact_dir=artifact_dir,
            max_features=tfidf_max_features,
            normalize_embeddings=normalize_embeddings,
            error_tracker=error_tracker,
        )
    if backend in {"sklearn-hashing", "hashing", "local-hashing"}:
        return HashingTextEmbedder(
            n_features=hashing_features,
            normalize_embeddings=normalize_embeddings,
            error_tracker=error_tracker,
        )
    if backend in {"auto", "sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            allow_fallback=allow_fallback,
            fallback_backend=fallback_backend,
            artifact_dir=artifact_dir,
            hashing_features=hashing_features,
            tfidf_max_features=tfidf_max_features,
            error_tracker=error_tracker,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


def embedder_from_vector_metadata(*, vector_dir: Path, error_tracker: ErrorTracker | None = None):
    """Create a query embedder that matches a previously built vector index."""
    metadata_path = Path(vector_dir) / VECTOR_METADATA_FILENAME
    if not metadata_path.exists():
        return create_embedder(artifact_dir=Path(vector_dir), error_tracker=error_tracker)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    backend = str(payload.get("embedding_backend") or "auto")
    model_name = payload.get("embedding_model")
    dim = int(payload.get("embedding_dim") or payload.get("hashing_features") or DEFAULT_HASHING_FEATURES)

    if backend == "tfidf" or str(model_name).startswith("sklearn-tfidf"):
        return TfidfTextEmbedder(
            artifact_dir=Path(vector_dir),
            vectorizer_path=Path(payload.get("tfidf_vectorizer_path") or Path(vector_dir) / TFIDF_VECTORIZER_FILENAME),
            max_features=int(payload.get("tfidf_max_features") or dim or DEFAULT_TFIDF_MAX_FEATURES),
            error_tracker=error_tracker,
        )

    if backend == "sklearn-hashing" or str(model_name).startswith("sklearn-hashing"):
        return HashingTextEmbedder(n_features=dim, error_tracker=error_tracker)

    return create_embedder(
        backend="sentence-transformers",
        model_name=model_name or DEFAULT_EMBEDDING_MODEL,
        hashing_features=dim,
        allow_fallback=False,
        artifact_dir=Path(vector_dir),
        error_tracker=error_tracker,
    )
