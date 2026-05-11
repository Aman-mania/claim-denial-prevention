#!/usr/bin/env python3
"""Download/cache the preferred Week 6 SentenceTransformer model once.

Run this before preferred Week 6 mode if your first ingestion attempt times out
while contacting Hugging Face.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def main() -> int:
    _load_dotenv(ROOT / ".env")
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", os.getenv("RAG_HF_ETAG_TIMEOUT", "60"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.getenv("RAG_HF_DOWNLOAD_TIMEOUT", "120"))

    print("Week 6 embedding model cache")
    print(f"  model:   {model_name}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. Run: python -m pip install -U sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    _ = model.encode(["policy retrieval smoke test"], normalize_embeddings=True)
    print("  ✓ model downloaded/cached and embedding smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
