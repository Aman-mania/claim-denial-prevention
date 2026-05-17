#!/usr/bin/env python3
"""Week 6/8 — Policy Document Ingestion and Vector Index Build.

Flow:
  policy docs (PDF/TXT/MD)
  → chunks with metadata
  → configured embeddings (OpenAI for Docker/AWS by default)
  → configured vector index (sklearn for Docker/AWS by default)
  → policy_chunks.parquet + policy_metadata.json + vector index artifact

Examples:
  # Docker/AWS semantic path, requires OPENAI_API_KEY
  python run_policy_ingest.py --embedding-backend openai --vector-backend sklearn --no-embedding-fallback

  # Local/offline fallback path
  python run_policy_ingest.py --embedding-backend tfidf --vector-backend sklearn
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from src.config import setup_logging
from src.io.table_store import LocalTableStore
from src.observability import ErrorCode, ErrorTracker, tracker_from_env
from src.rag.chunker import PolicyChunker
from src.rag.document_loader import PolicyDocumentLoader
from src.rag.embedder import DEFAULT_HASHING_FEATURES, DEFAULT_TFIDF_MAX_FEATURES, create_embedder
from src.rag.schemas import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_MODEL,
    POLICY_CHUNK_TABLE,
    POLICY_INGEST_REPORT_FILE,
)
from src.rag.vector_store import LocalVectorStore

BASE_DIR = Path(__file__).parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "policies" / "raw"
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "policies" / "processed"
DEFAULT_VECTOR_DIR = BASE_DIR / "data" / "vector_store"
SAMPLE_POLICY = BASE_DIR / "policy_docs" / "sample_claim_denial_policy_pack.md"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or str(value).strip() == "" else str(value).strip()


def _bootstrap_sample_policy(raw_dir: Path) -> bool:
    raw_dir.mkdir(parents=True, exist_ok=True)
    has_docs = any(path.is_file() and path.suffix.lower() in {".txt", ".md", ".markdown", ".pdf"} for path in raw_dir.rglob("*"))
    if has_docs:
        return False
    if SAMPLE_POLICY.exists():
        shutil.copyfile(SAMPLE_POLICY, raw_dir / SAMPLE_POLICY.name)
        return True
    return False


def parse_args() -> argparse.Namespace:
    default_embedding_backend = _env("RAG_EMBEDDING_BACKEND", "auto")
    default_vector_backend = _env("RAG_VECTOR_BACKEND", "sklearn")
    default_model = _env(
        "RAG_EMBEDDING_MODEL",
        _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") if default_embedding_backend == "openai" else DEFAULT_EMBEDDING_MODEL,
    )

    parser = argparse.ArgumentParser(description="Build the policy RAG index.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "openai", "sentence-transformers", "tfidf", "sklearn-hashing"],
        default=default_embedding_backend,
        help="Embedding backend. Docker/AWS recommended value: openai.",
    )
    parser.add_argument("--hashing-features", type=int, default=DEFAULT_HASHING_FEATURES)
    parser.add_argument("--tfidf-max-features", type=int, default=DEFAULT_TFIDF_MAX_FEATURES)
    parser.add_argument("--fallback-backend", choices=["tfidf", "sklearn-hashing"], default=_env("RAG_FALLBACK_BACKEND", "tfidf"))
    parser.add_argument(
        "--no-embedding-fallback",
        action="store_true",
        default=_env("RAG_ALLOW_EMBEDDING_FALLBACK", "true").lower() in {"0", "false", "no"},
        help="Fail instead of falling back when the selected embedding backend is unavailable.",
    )
    parser.add_argument(
        "--vector-backend",
        choices=["auto", "sklearn", "faiss", "numpy"],
        default=default_vector_backend,
        help="Vector index backend. Docker/AWS recommended value: sklearn.",
    )
    parser.add_argument("--chunk-size", type=int, default=int(_env("RAG_CHUNK_SIZE_WORDS", str(DEFAULT_CHUNK_SIZE_WORDS))))
    parser.add_argument("--overlap", type=int, default=int(_env("RAG_CHUNK_OVERLAP_WORDS", str(DEFAULT_CHUNK_OVERLAP_WORDS))))
    parser.add_argument("--no-bootstrap-sample", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level="INFO")
    tracker: ErrorTracker = tracker_from_env()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Policy RAG Ingestion            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        bootstrapped = False
        if not args.no_bootstrap_sample:
            bootstrapped = _bootstrap_sample_policy(args.raw_dir)

        loader = PolicyDocumentLoader(error_tracker=tracker)
        documents = loader.load_documents(args.raw_dir)

        chunker = PolicyChunker(
            chunk_size_words=args.chunk_size,
            chunk_overlap_words=args.overlap,
            embedding_model=args.model,
        )
        chunks = chunker.chunk_documents(documents)
        if not chunks:
            raise RuntimeError("No policy chunks were created. Check document content and chunk settings.")

        chunk_df = chunker.to_dataframe(chunks)
        chunk_store = LocalTableStore(args.processed_dir)
        chunk_path = chunk_store.write_table(POLICY_CHUNK_TABLE, chunk_df)

        embedder = create_embedder(
            backend=args.embedding_backend,
            model_name=args.model,
            allow_fallback=not args.no_embedding_fallback,
            hashing_features=args.hashing_features,
            tfidf_max_features=args.tfidf_max_features,
            fallback_backend=args.fallback_backend,
            artifact_dir=args.vector_dir,
            error_tracker=tracker,
        )
        embeddings = embedder.embed_texts(chunk_df["chunk_text"].tolist())
        embedding_metadata = embedder.metadata() if hasattr(embedder, "metadata") else {}

        metadata = chunk_df.to_dict(orient="records")
        index_info = LocalVectorStore(vector_dir=args.vector_dir, error_tracker=tracker, vector_backend=args.vector_backend).build(
            embeddings=embeddings,
            metadata=metadata,
            embedding_backend=embedding_metadata.get("embedding_backend"),
            embedding_model=embedding_metadata.get("embedding_model", args.model),
            embedding_metadata=embedding_metadata,
        )

        report = {
            "status": "success",
            "bootstrapped_sample_policy": bootstrapped,
            "raw_dir": str(args.raw_dir),
            "processed_dir": str(args.processed_dir),
            "vector_dir": str(args.vector_dir),
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "requested_embedding_model": args.model,
            "requested_embedding_backend": args.embedding_backend,
            "fallback_backend": args.fallback_backend,
            "embedding_backend": index_info.get("embedding_backend"),
            "embedding_model": index_info.get("embedding_model"),
            "embedding_dim": int(embeddings.shape[1]),
            "vector_backend": index_info.get("vector_backend"),
            "faiss_index_written": index_info.get("faiss_index_written"),
            "sklearn_index_written": index_info.get("sklearn_index_written"),
            "chunk_table_path": str(chunk_path),
            **index_info,
        }
        args.vector_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.vector_dir / POLICY_INGEST_REPORT_FILE
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        print(f"\n  Policy documents loaded: {len(documents):,}")
        print(f"  Policy chunks created:   {len(chunks):,}")
        print(f"  Embedding backend:       {index_info.get('embedding_backend')}")
        print(f"  Embedding model:         {index_info.get('embedding_model')}")
        print(f"  Vector backend:          {index_info.get('vector_backend')}")
        print(f"  Vector index:            {index_info['index_path']}")
        print(f"  Chunk table:             {chunk_path}")
        if bootstrapped:
            print("  Note: sample educational policy pack was copied because raw policy dir was empty.")
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   ✓  Policy RAG index built. Run: python run_policy_match.py ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 0

    except Exception as exc:
        event = tracker.record_exception(
            exc,
            component="rag",
            stage="policy_ingest_run",
            fallback_code=ErrorCode.RAG_UNEXPECTED,
            metadata={"stage": "policy_ingest_run"},
        )
        print(f"\n  ERROR: {exc}")
        print(f"  Error code: {event.error_code}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Policy RAG ingestion failed.                          ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1


if __name__ == "__main__":
    sys.exit(main())
