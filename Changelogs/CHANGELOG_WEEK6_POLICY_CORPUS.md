# Week 6 Policy Corpus Update

## Added

- `policy_docs/official_policy_source_registry.json` with CMS, HHS/HIPAA, payer, and internal policy sources.
- Curated RAG seed documents:
  - `policy_docs/rag_seed/us_healthcare_claim_policy_seed_pack.md`
  - `policy_docs/rag_seed/payer_policy_reference_pack.md`
  - `policy_docs/rag_seed/hipaa_operational_safeguards_pack.md`
- `tools/sync_policy_docs.py` to materialize source-controlled policy docs into `data/policies/raw/`.
- Registry tests under `tests/rag/test_policy_source_registry.py`.

## Usage

```bash
python tools/sync_policy_docs.py --include-curated --include-source-summaries
python run_policy_ingest.py
python run_policy_match.py
```

Optional government-source download:

```bash
python tools/sync_policy_docs.py --all
```

Private payer documents are summarized and referenced, not downloaded by default. Use `--include-payer-downloads` only after reviewing the payer site terms.
