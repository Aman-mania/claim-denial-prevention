#!/usr/bin/env python3
"""Idempotently adds the Week 6 Policy RAG tab to dev_dashboard/app.py.

This avoids overwriting your existing Week 4/Week 5 dashboard edits.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dev_dashboard" / "app.py"


def main() -> int:
    text = APP_PATH.read_text(encoding="utf-8")
    changed = False

    import_line = "from tabs.policy_rag import render_policy_rag_tab\n"
    if import_line not in text:
        anchor = "from tabs.ml_analysis import render_ml_tab\n"
        if anchor not in text:
            raise RuntimeError("Could not find ml_analysis import anchor in dev_dashboard/app.py")
        text = text.replace(anchor, anchor + import_line)
        changed = True

    if "📚 Policy RAG (Week 6)" not in text:
        if "tab_raw, tab_clean, tab_ml, tab_xai = st.tabs([" in text:
            text = text.replace(
                "tab_raw, tab_clean, tab_ml, tab_xai = st.tabs([",
                "tab_raw, tab_clean, tab_ml, tab_xai, tab_rag = st.tabs([",
            )
            text = text.replace(
                "    \"🧠 Explainable AI (Week 5)\",\n])",
                "    \"🧠 Explainable AI (Week 5)\",\n    \"📚 Policy RAG (Week 6)\",\n])",
            )
        elif "tab_raw, tab_clean, tab_ml = st.tabs([" in text:
            text = text.replace(
                "tab_raw, tab_clean, tab_ml = st.tabs([",
                "tab_raw, tab_clean, tab_ml, tab_rag = st.tabs([",
            )
            text = text.replace(
                "    \"🤖 ML Model (Week 4)\",\n])",
                "    \"🤖 ML Model (Week 4)\",\n    \"📚 Policy RAG (Week 6)\",\n])",
            )
        else:
            raise RuntimeError("Could not find dashboard tab declaration to patch")
        changed = True

    rag_block = """
with tab_rag:
    render_policy_rag_tab(root_dir=_ROOT, gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
"""
    if "render_policy_rag_tab(root_dir=_ROOT" not in text:
        text = text.rstrip() + "\n" + rag_block
        changed = True

    if changed:
        APP_PATH.write_text(text, encoding="utf-8")
        print(f"Updated {APP_PATH}")
    else:
        print("Week 6 dashboard integration already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
