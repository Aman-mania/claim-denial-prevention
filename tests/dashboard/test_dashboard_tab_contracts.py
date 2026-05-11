from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "dev_dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


def _accepts_kwargs(fn):
    sig = inspect.signature(fn)
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_policy_rag_exported_renderer_accepts_dashboard_kwargs():
    from tabs.policy_rag import render_policy_rag_tab

    assert callable(render_policy_rag_tab)
    assert _accepts_kwargs(render_policy_rag_tab) or "root_dir" in inspect.signature(render_policy_rag_tab).parameters


def test_explainability_exported_renderer_accepts_dashboard_kwargs():
    from tabs.explainability import render_explainability_tab

    assert callable(render_explainability_tab)
    assert _accepts_kwargs(render_explainability_tab) or "root_dir" in inspect.signature(render_explainability_tab).parameters


def test_backward_compatible_aliases_exist():
    import tabs.policy_rag as policy_rag
    import tabs.explainability as explainability

    assert policy_rag.render_policy_evidence_tab is policy_rag.render_policy_rag_tab
    assert explainability.render_risk_explanations_tab is explainability.render_explainability_tab
