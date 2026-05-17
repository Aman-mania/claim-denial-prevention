#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set before building OpenAI RAG vectors}"

rm -rf data/vector_store/* data/policies/processed/*
python run_policy_ingest.py \
  --embedding-backend openai \
  --vector-backend sklearn \
  --no-embedding-fallback

python scripts/check_openai_rag_ready.py
