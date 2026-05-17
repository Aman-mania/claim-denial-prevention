# OpenAI RAG + sklearn Vector Search Patch

## Purpose

This patch makes OpenAI embeddings the primary Docker/AWS RAG embedding backend and replaces the Docker/AWS vector-search path with a scikit-learn nearest-neighbor index. This avoids bundling `sentence-transformers`, FAISS, and PyTorch into the default Docker image.

## Runtime modes

| Environment | Embedding backend | Vector backend | Notes |
|---|---|---|---|
| Docker/AWS primary | `openai` | `sklearn` | Requires `OPENAI_API_KEY`; no Hugging Face/PyTorch/FAISS. |
| Local fallback | `tfidf` | `sklearn` | No internet or OpenAI key required. |
| Optional local semantic | `sentence-transformers` | `faiss` or `sklearn` | Install `requirements-semantic.txt`; not default. |

## Important data rule

Use OpenAI embeddings for policy documents and generic claim-reason queries only. Do not send raw claim payloads, patient identifiers, or PHI to OpenAI.

## Build OpenAI policy index

```bash
export OPENAI_API_KEY="sk-..."
export RAG_EMBEDDING_BACKEND=openai
export RAG_VECTOR_BACKEND=sklearn
export RAG_ALLOW_EMBEDDING_FALLBACK=false

bash scripts/build_openai_policy_index.sh
```

This regenerates:

```text
data/policies/processed/policy_chunks.parquet
data/vector_store/policy_vectors.npy
data/vector_store/policy_sklearn_nn.pkl
data/vector_store/policy_metadata.json
```

## Verify

```bash
python scripts/check_openai_rag_ready.py
```

Expected:

```text
OpenAI RAG readiness: PASS
embedding_backend=openai
vector_backend=sklearn
```

## Docker

```bash
cp .env.docker.example .env.docker
# edit OPENAI_API_KEY in .env.docker

docker compose down
docker compose build --no-cache
docker compose up -d
bash scripts/docker_smoke_test.sh
```

## AWS

Use `.env.aws.example` as the template for the EC2 `.env.docker` file. Set `APP_ENV=aws`, `AUTH_DATABASE_URL` to the RDS PostgreSQL URL, and `OPENAI_API_KEY` from a secure value.
