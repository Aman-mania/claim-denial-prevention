# Week 6 Runner Preflight Patch

Adds an explicit, configurable `run_week6.py` wrapper for local Policy RAG.

## Added
- `.env.example` with preferred/fallback/auto RAG settings.
- `run_week6.py` backend selection and preflight checks.
- `tools/cache_week6_embedding_model.py` to cache the SentenceTransformer model before ingestion.
- `requirements-week6-preferred.txt` for optional preferred local dependencies.

## Behavior
- Preferred mode: `sentence-transformers` + `faiss`.
- Fallback mode: `sklearn-hashing` + `numpy`.
- Preflight prints selected mode, model, vector backend, HF cache path, and dependency status.
- Preferred mode fails clearly if preferred dependencies are missing.
