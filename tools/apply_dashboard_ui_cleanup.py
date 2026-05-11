"""Apply non-destructive dashboard label cleanup.

This script only updates user-facing labels in dev_dashboard/app.py. It is safe to
run multiple times and does not change dashboard logic.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dev_dashboard" / "app.py"

REPLACEMENTS = {
    "🏥 Claim Denial Prevention — Dev Dashboard": "Claim Denial Prevention — Dev Dashboard",
    "📊 Raw Data (Bronze)": "Raw Data",
    "Raw Data (Bronze)": "Raw Data",
    "✨ Clean Data (Silver)": "Clean Data",
    "Clean Data (Silver)": "Clean Data",
    "🤖 ML Model (Week 4)": "Risk Model",
    "ML Model (Week 4)": "Risk Model",
    "🧠 Explainable AI (Week 5)": "Risk Explanations",
    "Explainable AI (Week 5)": "Risk Explanations",
    "📚 Policy RAG (Week 6)": "Policy Evidence",
    "Policy RAG (Week 6)": "Policy Evidence",
    "Run Week 5": "Generate Explanations",
    "Run Week 6": "Run Policy Retrieval",
    "Week 5": "Explainability",
    "Week 6": "Policy Retrieval",
}


def main() -> None:
    if not APP.exists():
        raise FileNotFoundError(f"Dashboard app not found: {APP}")

    text = APP.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    if text == original:
        print("No app.py label changes were needed.")
        return

    backup = APP.with_suffix(".py.bak_ui_cleanup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    APP.write_text(text, encoding="utf-8")
    print(f"Updated dashboard labels in {APP}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
