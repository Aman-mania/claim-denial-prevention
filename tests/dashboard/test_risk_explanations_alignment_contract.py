from pathlib import Path
import inspect

from dev_dashboard.tabs.explainability import render_explainability_tab


def test_render_explainability_accepts_dashboard_kwargs():
    sig = inspect.signature(render_explainability_tab)
    assert "root_dir" in sig.parameters
    assert "gold_dir" in sig.parameters
    assert "models_dir" in sig.parameters


def test_render_explainability_source_has_unique_widget_keys():
    source = Path("dev_dashboard/tabs/explainability.py").read_text()
    assert 'key="risk_explanations_level_filter"' in source
    assert 'key="risk_explanations_claim_select"' in source


def test_render_explainability_uses_aligned_full_width_layout():
    source = Path("dev_dashboard/tabs/explainability.py").read_text()
    assert "filter_col, select_col = st.columns" in source
    assert "_render_claim_header" in source
    assert "_render_reason_cards" in source
    assert "with st.container(border=True)" in source
    assert 'width="stretch"' in source
