#!/usr/bin/env python3
"""
Week 1 — Bronze Layer Ingestion Entry Point
=============================================
Runs the full ingestion pipeline:
  1. Validates and ingests all 4 raw CSVs into Bronze Parquet.
  2. Profiles each Bronze dataset and prints a data quality report.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.ingestion.ingest import BronzeIngestionPipeline
from src.ingestion.profiler import DataProfiler
from src.observability.pipeline_integration import (
    tracker_from_env,
    record_pipeline_report,
    record_exception_and_return_report,
    summarize_error_events,
)

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def _print_observability_summary(events) -> None:
    summary = summarize_error_events(events)
    if summary["errors_recorded"]:
        print(
            f"\n  Observability: recorded {summary['errors_recorded']} issue(s) "
            f"[codes: {', '.join(summary['codes'])}]"
        )
        if summary["repeated_errors"]:
            print(f"  Repeated issues detected: {summary['repeated_errors']}")


def main() -> int:
    """Returns exit code: 0 = success, 1 = one or more datasets failed."""
    setup_logging(level="INFO")
    tracker = tracker_from_env()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 1: Bronze Layer Ingestion  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        print("\n[ Step 1/2 ] Running ingestion pipeline …\n")

        pipeline = BronzeIngestionPipeline(raw_dir=RAW_DIR, bronze_dir=BRONZE_DIR)
        report = pipeline.run()
        obs_events = record_pipeline_report(
            report,
            component="ingestion",
            tracker=tracker,
            stage="run_ingestion",
        )

        any_failed = False
        print("\n  ┌─────────────────────────────────────────────────────────┐")
        print("  │  Ingestion Summary                                      │")
        print("  ├──────────────────┬──────────┬────────────────────────────┤")
        print(f"  │ {'Dataset':<16}  │ {'Rows':>8} │ {'Status':<26} │")
        print("  ├──────────────────┼──────────┼────────────────────────────┤")

        for name, result in report["datasets"].items():
            status = result["status"]
            rows = result.get("raw_rows", "—")
            val_flag = ""
            if result.get("validation", {}).get("status") == "warnings":
                val_flag = " (schema warnings)"
            if status == "failed":
                any_failed = True
            symbol = "✓" if status == "success" else "✗"
            print(f"  │ {symbol} {name:<15}│ {str(rows):>8} │ {status + val_flag:<26} │")

        print("  └──────────────────┴──────────┴────────────────────────────┘")

        print("\n[ Step 2/2 ] Running data profiler …\n")
        profiler = DataProfiler(bronze_dir=BRONZE_DIR)
        profiles = profiler.profile_all()
        profiler.print_report(profiles)

        _print_observability_summary(obs_events)

        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        if any_failed:
            print("║   ✗  Bronze layer incomplete — check errors above.          ║")
        else:
            print("║   ✓  Bronze layer complete. Ready for Week 2 analytics.     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        return 1 if any_failed else 0

    except Exception as exc:
        failure = record_exception_and_return_report(
            exc,
            component="ingestion",
            stage="run_ingestion",
            tracker=tracker,
        )
        print(f"\n  ERROR [{failure['error_code']}]: {failure['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
