# Dashboard UI cleanup and policy evidence de-duplication

## Changed
- Renamed UI labels to functionality-focused names instead of week-number labels.
- Replaced the Policy RAG tab with `Policy Evidence — Reason + Policy-Based Explanation`.
- Removed excessive status emojis from the policy evidence tab.
- Reworked claim-level policy display so reasons and policy evidence are shown once, using structured rows rather than repeated narrative sections.
- Added de-duplication of policy evidence rows by claim, reason, and policy chunk/source.
- Added a retrieval-quality summary by reason code.

## How to apply

```bash
unzip claim_denial_dashboard_ui_cleanup.zip -d /tmp/dashboard_ui_cleanup
rsync -av /tmp/dashboard_ui_cleanup/ ./
python tools/apply_dashboard_ui_cleanup.py
pytest tests/dashboard/test_policy_rag_ui_helpers.py -v
streamlit run dev_dashboard/app.py
```
