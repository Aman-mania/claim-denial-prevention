# Week 6 Vector Backend Fix

## Problem
Week 6 still failed when `faiss-cpu` was not installed. The prior patch only made the embedding backend resilient; the vector index layer still required FAISS.

## Fix
- `LocalFaissVectorStore` is now backend-aware and persists `policy_vectors.npy` for every build.
- `--vector-backend auto` uses FAISS if installed and NumPy brute-force inner product search otherwise.
- Retrieval loads the stored backend and can fall back from a FAISS-built index to NumPy if the vector matrix exists.
- The dependency checker now treats `sentence-transformers` and `faiss-cpu` as optional local quality/performance upgrades, not hard requirements.
- Retrieval no longer silently uses hashing queries against a SentenceTransformer-built index. If the index was built with SentenceTransformers, retrieval requires SentenceTransformers or a rebuild with `--embedding-backend sklearn-hashing`.

## Commands

```bash
python tools/check_week6_dependencies.py
rm -rf data/vector_store/*
python run_week6.py
```

Optional stronger mode:

```bash
python -m pip install -U sentence-transformers faiss-cpu
rm -rf data/vector_store/*
python run_policy_ingest.py --embedding-backend sentence-transformers --vector-backend faiss --no-embedding-fallback
python run_policy_match.py
```
