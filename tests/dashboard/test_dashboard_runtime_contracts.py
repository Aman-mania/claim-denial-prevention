from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "dev_dashboard"
for path in (ROOT, DASHBOARD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_dashboard_imports_can_resolve_src_and_tabs():
    from tabs.clean_data import render_clean_tab
    from tabs.explainability import render_explainability_tab
    from tabs.ml_analysis import render_ml_tab
    from tabs.policy_rag import render_policy_rag_tab
    from tabs.raw_data import render_raw_tab

    assert callable(render_raw_tab)
    assert callable(render_clean_tab)
    assert callable(render_ml_tab)
    assert callable(render_explainability_tab)
    assert callable(render_policy_rag_tab)


def test_newer_tabs_accept_dashboard_path_kwargs():
    from tabs.explainability import render_explainability_tab
    from tabs.policy_rag import render_policy_rag_tab

    for fn in (render_explainability_tab, render_policy_rag_tab):
        sig = inspect.signature(fn)
        params = sig.parameters
        assert "root_dir" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        assert "gold_dir" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        assert "models_dir" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_claim_selectboxes_have_unique_keys():
    policy_text = (DASHBOARD_DIR / "tabs" / "policy_rag.py").read_text()
    explain_text = (DASHBOARD_DIR / "tabs" / "explainability.py").read_text()
    assert 'key="policy_evidence_claim_select"' in policy_text
    assert 'key="risk_explanations_claim_select"' in explain_text


def test_updated_tabs_do_not_use_deprecated_width_argument():
    for rel in ["tabs/policy_rag.py", "tabs/explainability.py"]:
        text = (DASHBOARD_DIR / rel).read_text()
        assert "use_container_width" not in text
