"""Apply safe Streamlit dashboard hardening changes.

This script is intentionally conservative:
- replaces deprecated use_container_width=True/False across dev_dashboard with width="stretch"/"content";
- does not rewrite app logic or generated artifacts.

Run from repo root after applying dashboard patches:
    python tools/apply_dashboard_global_hardening.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dev_dashboard"

REPLACEMENTS = {
    "use_container_width=True": 'width="stretch"',
    "use_container_width=False": 'width="content"',
}


def main() -> int:
    changed: list[Path] = []
    for path in DASHBOARD_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in REPLACEMENTS.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed.append(path)
    if changed:
        print("Updated Streamlit width deprecations in:")
        for path in changed:
            print(f"  - {path.relative_to(ROOT)}")
    else:
        print("No deprecated use_container_width calls found in dev_dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
