#!/usr/bin/env python3
"""Idempotently add Week 6 dependencies to requirements.txt."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"
DEPS = [
    "sentence-transformers>=3.0,<6.0",
    "faiss-cpu>=1.8,<2.0",
    "pypdf>=4.0,<6.0",
]


def main() -> int:
    text = REQ.read_text(encoding="utf-8") if REQ.exists() else ""
    changed = False
    if "# ── Week 6: Policy RAG" not in text:
        text = text.rstrip() + "\n\n# ── Week 6: Policy RAG ───────────────────────────────────────────────────────\n"
        changed = True
    for dep in DEPS:
        package = dep.split(">=")[0]
        if package not in text:
            text += dep + "\n"
            changed = True
    if changed:
        REQ.write_text(text, encoding="utf-8")
        print(f"Updated {REQ}")
    else:
        print("Week 6 dependencies already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
