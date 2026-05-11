"""Apply dashboard-wide Streamlit compatibility hardening.

This script is intentionally small and idempotent. It replaces deprecated
``use_container_width`` calls in dev_dashboard/*.py files with the current
``width`` argument recommended by Streamlit.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dev_dashboard"

REPLACEMENTS = {
    "use_container_width=True": "width=\"stretch\"",
    "use_container_width = True": "width=\"stretch\"",
    "use_container_width=False": "width=\"content\"",
    "use_container_width = False": "width=\"content\"",
}


def harden_file(path: Path) -> bool:
    text = path.read_text()
    updated = text
    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)
    if updated != text:
        backup = path.with_suffix(path.suffix + ".bak_width")
        if not backup.exists():
            backup.write_text(text)
        path.write_text(updated)
        return True
    return False


def main() -> int:
    if not DASHBOARD_DIR.exists():
        print(f"Dashboard directory not found: {DASHBOARD_DIR}")
        return 1
    changed = []
    for path in DASHBOARD_DIR.rglob("*.py"):
        if harden_file(path):
            changed.append(path.relative_to(ROOT))
    if changed:
        print("Updated Streamlit width arguments in:")
        for item in changed:
            print(f"  - {item}")
    else:
        print("No deprecated use_container_width arguments found in dev_dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
