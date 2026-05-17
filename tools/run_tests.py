#!/usr/bin/env python3
"""Convenience runner for project test suites.

Examples:
    python tools/run_tests.py --suite week7
    python tools/run_tests.py --suite phase8
    python tools/run_tests.py --suite all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES: dict[str, list[str]] = {
    "week5": ["tests/explainability"],
    "week6": ["tests/rag"],
    "week7": ["tests/rules", "tests/agent", "tests/auth", "tests/api", "tests/product_ui"],
    "week8": ["tests/config", "tests/deployment"],
    "phase8": ["tests/config", "tests/deployment"],
    "rules": ["tests/rules"],
    "agent": ["tests/agent"],
    "auth": ["tests/auth"],
    "api": ["tests/api"],
    "product_ui": ["tests/product_ui"],
    "config": ["tests/config"],
    "deployment": ["tests/deployment"],
    "dashboard": ["tests/dashboard"],
    "explainability": ["tests/explainability"],
    "observability": ["tests/observability"],
    "inference": ["tests/inference"],
    "ml": ["tests/ml"],
    "rag": ["tests/rag"],
    "pipeline": ["tests/ingestion", "tests/silver", "tests/gold", "tests/ml", "tests/inference"],
    "all": ["tests"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped pytest suites.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="all")
    parser.add_argument("--fast", action="store_true", help="Exclude tests marked slow.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output.")
    parser.add_argument("--maxfail", type=int, default=None, help="Stop after N failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [sys.executable, "-m", "pytest", *SUITES[args.suite]]
    if args.fast:
        cmd.extend(["-m", "not slow"])
    if args.verbose:
        cmd.append("-v")
    if args.maxfail is not None:
        cmd.append(f"--maxfail={args.maxfail}")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
