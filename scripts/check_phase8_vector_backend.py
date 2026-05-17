#!/usr/bin/env python3
"""Check whether current vector artifacts match supported deployment backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "vector_store" / "policy_metadata.json"


def main() -> int:
    if not METADATA.exists():
        print(f"MISSING: {METADATA}")
        return 1
    payload = json.loads(METADATA.read_text(encoding="utf-8"))
    emb = payload.get("embedding_backend")
    vec = payload.get("vector_backend")
    model = payload.get("embedding_model")
    print("Vector artifact summary")
    print(f"  embedding_backend: {emb}")
    print(f"  embedding_model:   {model}")
    print(f"  vector_backend:    {vec}")
    print(f"  rows:              {payload.get('row_count')}")

    if emb == "openai" and vec == "sklearn":
        print("\nPASS: AWS/Docker primary OpenAI + sklearn artifacts are ready.")
        return 0
    if emb == "tfidf" and vec == "sklearn":
        print("\nPASS: Local fallback TF-IDF + sklearn artifacts are ready.")
        return 0
    print("\nWARNING: Artifacts do not match the recommended Phase 9 OpenAI/sklearn or local TF-IDF/sklearn modes.")
    print("Recommended AWS build:")
    print("  OPENAI_API_KEY=... bash scripts/build_openai_policy_index.sh")
    return 2


if __name__ == "__main__":
    sys.exit(main())
