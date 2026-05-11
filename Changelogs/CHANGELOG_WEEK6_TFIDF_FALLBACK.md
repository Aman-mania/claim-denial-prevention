# Week 6 TF-IDF Fallback Update

Adds a stronger offline fallback embedding backend for Week 6 RAG.

## Why

`sentence-transformers + faiss` remains the preferred local setup. However, when
Hugging Face is blocked or unavailable, the previous fallback used
HashingVectorizer. Hashing is stateless and robust, but weaker for a small fixed
policy corpus because it does not learn corpus vocabulary or IDF weights.

## What changed

- Added `tfidf` embedding backend.
- TF-IDF fits on policy chunks during ingestion and persists
  `data/vector_store/policy_tfidf_vectorizer.pkl`.
- Retrieval loads the same persisted vectorizer to avoid ingestion/query skew.
- Fallback mode in `run_week6.py` now uses `tfidf + numpy`.
- Hashing remains available as an emergency stateless fallback.

## Recommended local modes

Preferred semantic mode:

```bash
python run_week6.py --mode preferred
```

Offline fallback mode:

```bash
python run_week6.py --mode fallback --min-score 0.10
```

Explicit TF-IDF mode:

```bash
python run_policy_ingest.py --embedding-backend tfidf --vector-backend numpy
python run_policy_match.py --min-score 0.10
```
