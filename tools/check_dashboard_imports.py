"""Smoke-check dashboard imports and tab renderer contracts.

Run from the project root:
    python tools/check_dashboard_imports.py

This script adds both the repository root and dev_dashboard directory to sys.path.
That is required because dashboard tab modules import both `tabs.*` and `src.*`.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dev_dashboard"

for path in (ROOT, DASHBOARD_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tabs.clean_data import render_clean_tab  # noqa: E402
from tabs.explainability import render_explainability_tab  # noqa: E402
from tabs.ml_analysis import render_ml_tab  # noqa: E402
from tabs.policy_rag import render_policy_rag_tab  # noqa: E402
from tabs.raw_data import render_raw_tab  # noqa: E402


def _accepts_dashboard_paths(fn) -> bool:
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "root_dir" in params and "gold_dir" in params


def main() -> int:
    checks = {
        "render_raw_tab": render_raw_tab,
        "render_clean_tab": render_clean_tab,
        "render_ml_tab": render_ml_tab,
        "render_explainability_tab": render_explainability_tab,
        "render_policy_rag_tab": render_policy_rag_tab,
    }
    failures: list[str] = []
    for name, fn in checks.items():
        print(f"OK import: {name} -> {fn.__module__}.{fn.__name__}")
        if name in {"render_explainability_tab", "render_policy_rag_tab"} and not _accepts_dashboard_paths(fn):
            failures.append(f"{name} does not accept dashboard path kwargs")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Dashboard import contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
