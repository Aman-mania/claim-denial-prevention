#!/usr/bin/env python3
"""
Week 3 — Silver Layer Cleaning Entry Point
============================================
Runs the Silver cleaning pipeline: Bronze Parquet → cleaned Silver Parquet.

Usage:
    python run_silver.py

Requires Bronze layer (run_ingestion.py) to exist first.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.silver.clean import SilverCleaningPipeline

BASE_DIR   = Path(__file__).parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"
SILVER_DIR = BASE_DIR / "data" / "silver"


def main() -> int:
    setup_logging(level="INFO")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 3: Silver Layer Cleaning   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    pipeline = SilverCleaningPipeline(bronze_dir=BRONZE_DIR, silver_dir=SILVER_DIR)
    report   = pipeline.run()

    any_failed = False

    print("\n  ┌──────────────────┬──────────┬──────────┬────────────────────┐")
    print(f"  │ {'Dataset':<16}  │ {'Bronze':>8} │ {'Silver':>8} │ {'Status':<18} │")
    print("  ├──────────────────┼──────────┼──────────┼────────────────────┤")

    for name, result in report["datasets"].items():
        status = result["status"]
        if status == "failed":
            any_failed = True
        bronze_r = result.get("bronze_rows", "—")
        silver_r = result.get("silver_rows", "—")
        removed  = result.get("rows_removed", 0)
        symbol   = "✓" if status == "success" else "✗"
        removed_str = f"-{removed} dupes" if removed else "no dupes"
        print(f"  │ {symbol} {name:<15}│ {str(bronze_r):>8} │ {str(silver_r):>8} │ {removed_str:<18} │")

    print("  └──────────────────┴──────────┴──────────┴────────────────────┘")

    # Validation summary
    print("\n  Validation results:")
    for name, result in report["datasets"].items():
        val = result.get("validation", {})
        status = val.get("status", "—")
        symbol = "✓" if status == "passed" else "⚠" if status == "warnings" else "—"
        print(f"    {symbol}  {name:<12}  {status}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    if any_failed:
        print("║   ✗  Silver pipeline incomplete — check errors above.       ║")
    else:
        print("║   ✓  Silver layer complete. Ready for Week 4 Gold layer.    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
