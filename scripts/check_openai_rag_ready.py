#!/usr/bin/env python3
"""Check whether the project is ready for OpenAI-embedding RAG deployment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "data" / "vector_store"
METADATA_PATH = VECTOR_DIR / "policy_metadata.json"


def _ok(label: str) -> None:
    print(f"  OK       {label}")


def _warn(label: str) -> None:
    print(f"  WARNING  {label}")


def _fail(label: str) -> None:
    print(f"  MISSING  {label}")


def main() -> int:
    print("OpenAI RAG readiness check")
    exit_code = 0

    if os.getenv("OPENAI_API_KEY"):
        _ok("OPENAI_API_KEY is set")
    else:
        _fail("OPENAI_API_KEY is not set")
        exit_code = 1

    try:
        import openai  # noqa: F401
        _ok("openai package import works")
    except Exception as exc:
        _fail(f"openai package import failed: {exc}")
        exit_code = 1

    if not METADATA_PATH.exists():
        _fail(f"vector metadata missing: {METADATA_PATH}")
        print("\nBuild OpenAI policy vectors with:")
        print("  python run_policy_ingest.py --embedding-backend openai --vector-backend sklearn --no-embedding-fallback")
        return 1

    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    embedding_backend = payload.get("embedding_backend")
    vector_backend = payload.get("vector_backend")
    embedding_model = payload.get("embedding_model")
    row_count = payload.get("row_count")

    print(f"  metadata: embedding_backend={embedding_backend}, embedding_model={embedding_model}, vector_backend={vector_backend}, rows={row_count}")

    if embedding_backend == "openai":
        _ok("policy vectors were built with OpenAI embeddings")
    else:
        _fail("policy vectors were not built with OpenAI embeddings")
        exit_code = 1

    if vector_backend == "sklearn":
        _ok("policy index uses sklearn vector search")
    else:
        _fail("policy index does not use sklearn vector search")
        exit_code = 1

    if payload.get("sklearn_index_written") or (VECTOR_DIR / "policy_sklearn_nn.pkl").exists():
        _ok("sklearn nearest-neighbor index exists")
    else:
        _warn("sklearn index pickle not found; runtime can rebuild from policy_vectors.npy if matrix exists")

    matrix = VECTOR_DIR / "policy_vectors.npy"
    if matrix.exists():
        _ok("policy_vectors.npy exists")
    else:
        _fail("policy_vectors.npy missing")
        exit_code = 1

    if exit_code == 0:
        print("\nOpenAI RAG readiness: PASS")
    else:
        print("\nOpenAI RAG readiness: FAIL")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
