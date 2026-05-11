# Retrieval Analytics Dashboard Tab

Adds a new Streamlit tab for visual diagnostics of the explainability + policy retrieval layer.

## Added

- `dev_dashboard/tabs/retrieval_analytics.py`
- `tools/apply_retrieval_analytics_tab.py`
- `tools/check_retrieval_analytics_tab.py`
- `tests/dashboard/test_retrieval_analytics_tab.py`

## Visuals

- Retrieval overview metrics
- Policy vector-space scatter with PCA/t-SNE projection
- Selected-claim policy chunk highlighting
- Similarity score histogram
- Similarity by reason
- Reason-to-policy Sankey flow
- Selected-claim retrieval graph
- Top retrieved policy chunks
- SHAP contribution vs policy similarity scatter
- Retrieval table explorer

## Notes

- The tab is read-only and does not change pipeline outputs.
- It uses existing artifacts from Week 5/6.
- It accepts the common dashboard renderer signature:
  `render_retrieval_analytics_tab(root_dir=None, gold_dir=None, models_dir=None, **kwargs)`.
- `width="stretch"` is used instead of deprecated `use_container_width=True`.
