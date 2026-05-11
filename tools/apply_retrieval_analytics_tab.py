
#!/usr/bin/env python3
"""
Idempotently add the Retrieval Analytics tab to dev_dashboard/app.py.

The script intentionally uses conservative text edits and keeps a backup:
dev_dashboard/app.py.bak_retrieval_analytics
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dev_dashboard" / "app.py"
BACKUP_PATH = ROOT / "dev_dashboard" / "app.py.bak_retrieval_analytics"


IMPORT_LINE = "from tabs.retrieval_analytics import render_retrieval_analytics_tab"
TAB_LABEL = "Retrieval Analytics"
TAB_VAR = "tab_retrieval_analytics"


def _insert_import(text: str) -> str:
    if IMPORT_LINE in text:
        return text

    import_lines = list(re.finditer(r"^from tabs\.[^\n]+$", text, flags=re.MULTILINE))
    if import_lines:
        last = import_lines[-1]
        return text[: last.end()] + "\n" + IMPORT_LINE + text[last.end() :]

    # Fallback: place after any import block.
    return IMPORT_LINE + "\n" + text


def _find_tabs_assignment(text: str) -> re.Match[str] | None:
    # Handles common Streamlit pattern:
    # tab_a, tab_b = st.tabs(["A", "B"])
    pattern = re.compile(
        r"(?P<lhs>^[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*[ \t]*,[ \t]*)+[A-Za-z_][A-Za-z0-9_]*[ \t]*)=[ \t]*st\.tabs\(\[(?P<labels>.*?)\]\)",
        flags=re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        labels = match.group("labels")
        if "Raw Data" in labels or "Risk Model" in labels or "Policy Evidence" in labels or "ML Model" in labels:
            return match
    return None


def _insert_tab_assignment(text: str) -> str:
    if TAB_LABEL in text and TAB_VAR in text:
        return text

    match = _find_tabs_assignment(text)
    if not match:
        raise RuntimeError(
            "Could not locate st.tabs assignment in dev_dashboard/app.py. "
            "Add the tab manually by importing render_retrieval_analytics_tab, adding "
            f"'{TAB_LABEL}' to st.tabs(...), and rendering it under a new with-block."
        )

    lhs = match.group("lhs").rstrip()
    labels = match.group("labels").rstrip()

    new_lhs = lhs
    if TAB_VAR not in lhs:
        new_lhs = lhs + ", " + TAB_VAR

    new_labels = labels
    if TAB_LABEL not in labels:
        sep = "" if labels.strip().endswith(",") else ","
        new_labels = labels + sep + f'\n    "{TAB_LABEL}"'

    replacement = f"{new_lhs} = st.tabs([{new_labels}])"
    return text[: match.start()] + replacement + text[match.end() :]


def _root_arg(text: str) -> str:
    if "_ROOT" in text:
        return "_ROOT"
    if "ROOT" in text:
        return "ROOT"
    return "Path.cwd()"


def _gold_arg(text: str) -> str:
    return "GOLD_DIR" if "GOLD_DIR" in text else "None"


def _models_arg(text: str) -> str:
    return "MODELS_DIR" if "MODELS_DIR" in text else "None"


def _insert_render_block(text: str) -> str:
    if "render_retrieval_analytics_tab(" in text:
        return text

    root = _root_arg(text)
    gold = _gold_arg(text)
    models = _models_arg(text)
    block = (
        f"\nwith {TAB_VAR}:\n"
        f"    render_retrieval_analytics_tab(root_dir={root}, gold_dir={gold}, models_dir={models})\n"
    )

    # Prefer placing after the policy evidence/rag block.
    policy_match = None
    for match in re.finditer(r"^with\s+tab_[A-Za-z0-9_]*rag[A-Za-z0-9_]*\s*:\s*\n(?:^[ \t]+.*\n)+", text, flags=re.MULTILINE):
        policy_match = match
    if policy_match:
        return text[: policy_match.end()] + block + text[policy_match.end() :]

    # Otherwise append to end.
    return text.rstrip() + "\n" + block + "\n"


def main() -> int:
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Could not find {APP_PATH}")

    text = APP_PATH.read_text()
    updated = _insert_import(text)
    updated = _insert_tab_assignment(updated)
    updated = _insert_render_block(updated)

    if updated == text:
        print("Retrieval Analytics tab is already configured.")
        return 0

    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(text)
    APP_PATH.write_text(updated)
    print(f"Updated {APP_PATH}")
    print(f"Backup: {BACKUP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
