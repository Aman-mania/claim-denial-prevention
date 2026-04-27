#!/usr/bin/env python3
"""
Week 1 — Bronze Layer Ingestion Entry Point
=============================================
Runs the full ingestion pipeline:
  1. Validates and ingests all 4 raw CSVs into Bronze Parquet.
  2. Profiles each Bronze dataset and prints a data quality report.

Usage
-----
    python run_ingestion.py

Expects
-------
    data/raw/claims_1000.csv
    data/raw/providers_1000.csv
    data/raw/diagnosis.csv
    data/raw/cost.csv

Outputs
-------
    data/bronze/claims/claims_bronze.parquet
    data/bronze/providers/providers_bronze.parquet
    data/bronze/diagnosis/diagnosis_bronze.parquet
    data/bronze/cost/cost_bronze.parquet
    logs/ingestion.log  (if file handler added — see src/config.py)
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.ingestion.ingest import BronzeIngestionPipeline
from src.ingestion.profiler import DataProfiler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def main() -> int:
    """Returns exit code: 0 = success, 1 = one or more datasets failed."""
    setup_logging(level="INFO")

    print()
    print("║   Claim Denial Prevention — Week 1: Bronze Layer Ingestion  ║")

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    print("\n[ Step 1/2 ] Running ingestion pipeline …\n")

    pipeline = BronzeIngestionPipeline(raw_dir=RAW_DIR, bronze_dir=BRONZE_DIR)
    report = pipeline.run()

    any_failed = False
    print("  │  Ingestion Summary                                      │")
    print(f"  │ {'Dataset':<16}  │ {'Rows':>8} │ {'Status':<26} │")

    for name, result in report["datasets"].items():
        status   = result["status"]
        rows     = result.get("raw_rows", "—")
        val_flag = ""
        if result.get("validation", {}).get("status") == "warnings":
            val_flag = " (schema warnings)"
        if status == "failed":
            any_failed = True
        symbol = "✓" if status == "success" else "✗"
        print(f"  │ {symbol} {name:<15}│ {str(rows):>8} │ {status + val_flag:<26} │")

    # ── Step 2: Profile ────────────────────────────────────────────────────────
    print("\n[ Step 2/2 ] Running data profiler …\n")

    profiler = DataProfiler(bronze_dir=BRONZE_DIR)
    profiles = profiler.profile_all()
    profiler.print_report(profiles)

    # ── Done ───────────────────────────────────────────────────────────────────
    print()
    if any_failed:
        print("║   ✗  Bronze layer incomplete — check errors above.          ║")
    else:
        print("║   ✓  Bronze layer complete. Ready for Week 2 analytics.     ║")
    print()

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
