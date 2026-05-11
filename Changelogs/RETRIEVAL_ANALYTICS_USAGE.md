
# Retrieval Analytics Tab Usage

Apply the patch, then run:

```bash
python tools/apply_retrieval_analytics_tab.py
python -m py_compile dev_dashboard/tabs/retrieval_analytics.py tools/apply_retrieval_analytics_tab.py tools/check_retrieval_analytics_tab.py
python tools/check_retrieval_analytics_tab.py
pytest tests/dashboard/test_retrieval_analytics_tab.py -v
streamlit run dev_dashboard/app.py
```

Required generated artifacts:

```bash
python run_explain.py
python run_policy_ingest.py
python run_policy_match.py
```

or:

```bash
python run_week6.py --mode preferred
```
