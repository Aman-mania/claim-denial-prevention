# Policy Documents for Week 6 RAG

This folder contains source-controlled policy seed documents and a source registry for the Week 6 RAG layer.

## Why these files are source-controlled

`data/policies/raw/` is intentionally gitignored because downloaded PDFs, HTML pages, and generated artifacts can become large. The source-controlled corpus lives here under `policy_docs/`, then `tools/sync_policy_docs.py` materializes it into `data/policies/raw/` for local ingestion.

## Recommended flow

```bash
python tools/sync_policy_docs.py --include-curated --include-source-summaries
python run_policy_ingest.py
python run_policy_match.py
```

To also download public government source pages/PDFs listed in the registry:

```bash
python tools/sync_policy_docs.py --include-curated --include-source-summaries --download-official
```

Private payer policy pages are included as reference metadata and summarized in curated seed documents. They are **not downloaded by default** because payer pages may have site-specific terms and copyright restrictions. Use `--include-payer-downloads` only if you have reviewed and accepted the relevant payer terms for local research/development use.

## Files

- `official_policy_source_registry.json`: machine-readable registry of government, payer, and internal policy sources.
- `rag_seed/us_healthcare_claim_policy_seed_pack.md`: curated US healthcare claim-denial policy pack for reliable RAG testing.
- `rag_seed/payer_policy_reference_pack.md`: payer policy reference summaries and source links.
- `rag_seed/hipaa_operational_safeguards_pack.md`: HIPAA privacy/security/breach safeguards relevant to logging, dashboarding, and future API/agent work.
