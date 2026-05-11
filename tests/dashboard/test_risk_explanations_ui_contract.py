from __future__ import annotations

import ast
from pathlib import Path

MODULE_PATH = Path("dev_dashboard/tabs/explainability.py")


def _source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_risk_explanation_renderer_accepts_shared_dashboard_kwargs() -> None:
    module = ast.parse(_source())
    funcs = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    assert "render_explainability_tab" in funcs
    func = funcs["render_explainability_tab"]
    arg_names = [arg.arg for arg in func.args.args]
    assert "root_dir" in arg_names
    assert "gold_dir" in arg_names
    assert "models_dir" in arg_names
    assert func.args.kwarg is not None


def test_risk_explanation_widgets_have_unique_keys() -> None:
    src = _source()
    assert 'key="risk_explanations_level_filter"' in src
    assert 'key="risk_explanations_claim_select"' in src


def test_risk_explanation_ui_no_internal_week_labels() -> None:
    src = _source()
    assert 'st.header("Week' not in src
    assert "Week 5" not in src
    assert "Week 6" not in src


def test_risk_explanation_uses_structured_claim_layout() -> None:
    src = _source()
    assert "left, right = st.columns([1, 2.3])" in src
    assert "_display_reason_card" in src
    assert "with st.container(border=True)" in src
