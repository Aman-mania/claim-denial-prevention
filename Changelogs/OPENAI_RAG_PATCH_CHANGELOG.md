# Changelog — OpenAI RAG Docker/AWS Patch

## Added
- `OpenAITextEmbedder` in `src/rag/embedder.py`.
- `sklearn` vector backend in `src/rag/vector_store.py`.
- OpenAI-first Docker/AWS env examples.
- Lightweight Docker requirements that include `openai` but exclude `sentence-transformers` and `faiss-cpu`.
- Readiness scripts for OpenAI RAG artifacts.
- Tests for OpenAI embedder and sklearn vector store.

## Changed
- `run_policy_ingest.py` now accepts `--embedding-backend openai` and `--vector-backend sklearn`.
- Docker defaults now target `RAG_EMBEDDING_BACKEND=openai` and `RAG_VECTOR_BACKEND=sklearn`.

## Preserved
- TF-IDF and hashing remain available for local fallback.
- Sentence-transformers and FAISS remain optional but are not part of the default Docker image.
