# Retrieval Analytics Source Visibility Fix

## Problem
Some policy sources were present in `policy_chunks.parquet` and visible in tables, but did not appear clearly in multiple visuals because several charts were driven only from `gold_claim_policy_matches.parquet`. That means sources with zero or low retrieval counts were excluded from retrieval-only charts. In the source coverage chart, tiny corpus counts were also plotted beside very large evidence counts, making corpus bars nearly invisible.

## Fix
- Rebuilt `dev_dashboard/tabs/retrieval_analytics.py` to always distinguish:
  - policy corpus coverage (`policy_chunks.parquet`)
  - retrieval usage (`gold_claim_policy_matches.parquet`)
- Added reusable source-presence visuals across tabs where retrieval-only views could hide sources.
- Updated source coverage to use separate charts for corpus chunks and retrieved evidence rows instead of overlaying incomparable scales.
- Updated vector scatter to use high-contrast fixed source colors, marker borders, larger markers, and selected-claim overlays.
- Updated reason-policy and claim-level Sankey diagrams to include all corpus source nodes and to color nodes/links by source.
- Updated score quality, reason-policy flow, claim graph, SHAP-vs-policy, and top-chunk sections to show source coverage context.
- Kept unique Streamlit keys, `width="stretch"`, and stable dashboard renderer signatures.

## Verification
Run:

```bash
python -m py_compile dev_dashboard/tabs/retrieval_analytics.py
pytest tests/dashboard/test_retrieval_analytics_full_visibility.py -v
streamlit run dev_dashboard/app.py
```
