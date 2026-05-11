from __future__ import annotations

import importlib.util
from pathlib import Path


def test_policy_rag_tab_exports_expected_entrypoint():
    path = Path("dev_dashboard/tabs/policy_rag.py")
    spec = importlib.util.spec_from_file_location("policy_rag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "render_policy_rag_tab")
    assert callable(module.render_policy_rag_tab)
    assert module.render_policy_evidence_tab is module.render_policy_rag_tab
