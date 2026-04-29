#!/usr/bin/env python3
"""
Week 4 — Gold Feature Pipeline Entry Point
===========================================
Silver → Gold base table → Gold feature table → feature_manifest.json

Usage:
    python run_gold.py

Requires Silver layer (run_silver.py) to exist first.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.gold.features import GoldFeaturePipeline

BASE_DIR   = Path(__file__).parent
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR   = BASE_DIR / "data" / "gold"


def main() -> int:
    setup_logging(level="INFO")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 4: Gold Feature Pipeline   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    pipeline = GoldFeaturePipeline(silver_dir=SILVER_DIR, gold_dir=GOLD_DIR)
    report   = pipeline.run()

    if report["status"] == "success":
        print(f"\n  Base table rows:    {report['base_rows']:,}")
        print(f"  Feature table rows: {report['feature_rows']:,}")
        print(f"  Denied claims:      {report['denied_count']:,} ({report['denial_rate_pct']}%)")
        print(f"  Approved claims:    {report['base_rows'] - report['denied_count']:,}")
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✓  Gold layer complete. Run: python run_train.py          ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 0
    else:
        print(f"\n  ERROR: {report.get('error')}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Gold pipeline failed.                                  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1


if __name__ == "__main__":
    sys.exit(main())
