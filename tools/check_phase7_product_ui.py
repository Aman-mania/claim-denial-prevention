#!/usr/bin/env python3
"""Preflight checks for the role-aware Streamlit product UI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_status(name: str) -> str:
    return "OK" if importlib.util.find_spec(name) else "MISSING"


def main() -> int:
    print("Phase 7 product UI preflight")
    print(f"  streamlit: {_module_status('streamlit')}")
    print(f"  requests:  {_module_status('requests')}")
    print(f"  fastapi:   {_module_status('fastapi')} / needed for backend, not UI import")
    print(f"  API URL:   {os.getenv('CLAIM_DENIAL_API_BASE_URL', 'http://localhost:8000')}")
    print("  Files:")
    required = [
        ROOT / "product_ui" / "app.py",
        ROOT / "product_ui" / "api_client.py",
        ROOT / "product_ui" / "rendering.py",
        ROOT / "api" / "main.py",
        ROOT / "src" / "agent" / "remediation_agent.py",
    ]
    missing = False
    for path in required:
        ok = path.exists()
        missing = missing or not ok
        print(f"    {'OK     ' if ok else 'MISSING'} {path.relative_to(ROOT)}")
    if missing:
        return 1
    print("\nStart backend first:")
    print("  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    print("\nThen start product UI:")
    print("  streamlit run product_ui/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
