# Week 6 Embedding Backend Fallback Fix

## Problem

`python run_week6.py` failed during `run_policy_ingest.py` when the active virtual environment did not have the `sentence_transformers` import available:

```text
No module named 'sentence_transformers'
Error code: RAG_003
```

This prevented policy chunks from being embedded and blocked vector index creation.

## Fix

- Added `HashingTextEmbedder`, a deterministic scikit-learn `HashingVectorizer` fallback.
- Updated `SentenceTransformerEmbedder` to prefer Sentence Transformers but automatically fall back to hashing when allowed.
- Added embedding backend metadata to `policy_metadata.json` so policy matching uses the same backend/dimension used during index creation.
- Added `--embedding-backend` and `--no-embedding-fallback` options to `run_policy_ingest.py`.
- Added `tools/check_week6_dependencies.py` to verify the active Python environment.
- Added tests for the fallback embedder.

## Recommended local command

```bash
python run_policy_ingest.py --embedding-backend auto
```

If Sentence Transformers is unavailable, it will continue with:

```text
sklearn-hashing-4096
```

For strict semantic embeddings, install Sentence Transformers in the same venv:

```bash
python -m pip install -U sentence-transformers
python run_policy_ingest.py --embedding-backend sentence-transformers --no-embedding-fallback
```
