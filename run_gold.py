#!/usr/bin/env python3
"""
Week 4 — Gold Feature Pipeline Entry Point
===========================================
Silver → Gold base table → Gold feature table → feature_manifest.json →
inference_artifacts.json.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.gold.features import GoldFeaturePipeline
from src.observability.pipeline_integration import (
    tracker_from_env,
    record_pipeline_report,
    record_exception_and_return_report,
    summarize_error_events,
)

BASE_DIR = Path(__file__).parent
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"


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
    print("║   Claim Denial Prevention — Week 4: Gold Feature Pipeline   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        pipeline = GoldFeaturePipeline(silver_dir=SILVER_DIR, gold_dir=GOLD_DIR)
        report = pipeline.run()
        obs_events = record_pipeline_report(
            report,
            component="gold",
            tracker=tracker,
            stage="run_gold",
        )

        if report["status"] == "success":
            print(f"\n  Base table rows:    {report['base_rows']:,}")
            print(f"  Feature table rows: {report['feature_rows']:,}")
            print(f"  Denied claims:      {report['denied_count']:,} ({report['denial_rate_pct']}%)")
            print(f"  Approved claims:    {report['base_rows'] - report['denied_count']:,}")
            print(f"  Label source:       {report.get('label_source', 'unknown')}")
            if report.get("cost_match_counts"):
                print(f"  Cost matches:       {report['cost_match_counts']}")
            _print_observability_summary(obs_events)
            print()
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║   ✓  Gold layer complete. Run: python run_train.py          ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            return 0

        print(f"\n  ERROR: {report.get('error')}")
        _print_observability_summary(obs_events)
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Gold pipeline failed.                                  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1

    except Exception as exc:
        failure = record_exception_and_return_report(
            exc,
            component="gold",
            stage="run_gold",
            tracker=tracker,
        )
        print(f"\n  ERROR [{failure['error_code']}]: {failure['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
