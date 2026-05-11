
#!/usr/bin/env python3
"""Check dashboard tab import contracts including Retrieval Analytics."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dev_dashboard"

for path in [str(ROOT), str(DASHBOARD)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def _check_renderer(module_name: str, function_name: str) -> None:
    module = __import__(module_name, fromlist=[function_name])
    renderer = getattr(module, function_name)
    sig = inspect.signature(renderer)
    for name in ["root_dir", "gold_dir", "models_dir"]:
        if name not in sig.parameters:
            raise AssertionError(f"{module_name}.{function_name} missing parameter {name}")


def main() -> int:
    _check_renderer("tabs.retrieval_analytics", "render_retrieval_analytics_tab")
    _check_renderer("tabs.policy_rag", "render_policy_rag_tab")
    _check_renderer("tabs.explainability", "render_explainability_tab")
    print("Dashboard analytics import contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
