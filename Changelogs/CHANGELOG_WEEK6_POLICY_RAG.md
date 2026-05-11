# Week 6 Policy RAG Implementation

## Added

- Local policy document loader for TXT/MD/PDF policy sources.
- Metadata-preserving policy chunker with deterministic policy tags.
- SentenceTransformers embedding wrapper.
- Local FAISS vector store abstraction.
- Reason-aware policy retriever.
- Policy matcher that consumes `gold_claim_explanations.parquet` and writes:
  - `gold_claim_policy_matches.parquet`
  - `gold_claim_final_explanations.parquet`
- Separate Streamlit tab module for Week 6 Policy RAG.
- Sample educational policy pack for local demos.
- Week 6 tests for document loading, chunking, vector store, policy matching, and table contracts.

## Commands

```bash
python tools/apply_week6_requirements.py
pip install -r requirements.txt
python run_explain.py
python run_policy_ingest.py
python run_policy_match.py
# or
python run_week6.py
```

Add dashboard tab integration:

```bash
python tools/apply_week6_dashboard_integration.py
streamlit run dev_dashboard/app.py
```

## Cloud-readiness

Local FAISS and Parquet are intentionally hidden behind small boundaries:

- `LocalFaissVectorStore` can later be replaced by Databricks Vector Search.
- `LocalTableStore` can later be replaced by a Delta/Unity Catalog implementation.
- `SentenceTransformerEmbedder` can later be replaced by a Databricks Model Serving embedding endpoint.

## Important

The included policy pack is a synthetic educational corpus. It is meant for local
pipeline testing and dashboard demos only. Replace or supplement it with official
payer/CMS/internal policy documents for a real deployment.
