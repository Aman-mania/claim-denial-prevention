#!/usr/bin/env python3
"""
Week 3 — Silver Layer Cleaning Entry Point
============================================
Runs the Silver cleaning pipeline: Bronze Parquet → cleaned Silver Parquet.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.silver.clean import SilverCleaningPipeline
from src.observability.pipeline_integration import (
    tracker_from_env,
    record_pipeline_report,
    record_exception_and_return_report,
    summarize_error_events,
)

BASE_DIR = Path(__file__).parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"
SILVER_DIR = BASE_DIR / "data" / "silver"


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
    setup_logging(level="INFO")
    tracker = tracker_from_env()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 3: Silver Layer Cleaning   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        pipeline = SilverCleaningPipeline(bronze_dir=BRONZE_DIR, silver_dir=SILVER_DIR)
        report = pipeline.run()
        obs_events = record_pipeline_report(
            report,
            component="silver",
            tracker=tracker,
            stage="run_silver",
        )

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
            removed = result.get("rows_removed", 0)
            symbol = "✓" if status == "success" else "✗"
            removed_str = f"-{removed} dupes" if removed else "no dupes"
            print(f"  │ {symbol} {name:<15}│ {str(bronze_r):>8} │ {str(silver_r):>8} │ {removed_str:<18} │")

        print("  └──────────────────┴──────────┴──────────┴────────────────────┘")

        print("\n  Validation results:")
        for name, result in report["datasets"].items():
            val = result.get("validation", {})
            status = val.get("status", "—")
            symbol = "✓" if status == "passed" else "⚠" if status == "warnings" else "—"
            print(f"    {symbol}  {name:<12}  {status}")

        _print_observability_summary(obs_events)

        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        if any_failed:
            print("║   ✗  Silver pipeline incomplete — check errors above.       ║")
        else:
            print("║   ✓  Silver layer complete. Ready for Week 4 Gold layer.    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        return 1 if any_failed else 0

    except Exception as exc:
        failure = record_exception_and_return_report(
            exc,
            component="silver",
            stage="run_silver",
            tracker=tracker,
        )
        print(f"\n  ERROR [{failure['error_code']}]: {failure['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
