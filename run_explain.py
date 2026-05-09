#!/usr/bin/env python3
"""
Week 5 — Explainable AI Entry Point
===================================
Gold features + trained model + SHAP → business reasons table.

Usage:
    python run_explain.py
    python run_explain.py --limit 100
    python run_explain.py --claim-id C0001 --claim-id C0002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import setup_logging
from src.explainability import ExplanationGenerationPipeline

BASE_DIR = Path(__file__).parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week 5 claim explanations.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of claims to explain.")
    parser.add_argument("--claim-id", action="append", dest="claim_ids", help="Explain one or more specific claim IDs.")
    parser.add_argument("--max-reasons", type=int, default=3, help="Max business reasons per claim.")
    parser.add_argument("--shap-top-n", type=int, default=10, help="Top SHAP contributors to inspect before mapping.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level="INFO")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 5: Explainable AI          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    pipeline = ExplanationGenerationPipeline(
        gold_dir=GOLD_DIR,
        models_dir=MODELS_DIR,
        output_dir=GOLD_DIR,
        max_reasons=args.max_reasons,
        shap_top_n=args.shap_top_n,
    )
    report = pipeline.run(limit=args.limit, claim_ids=args.claim_ids)

    try:
        from src.observability.pipeline_integration import (
            record_pipeline_report,
            summarize_error_events,
            tracker_from_env,
        )
        tracker = tracker_from_env()
        events = record_pipeline_report(report, component="ml", tracker=tracker, stage="run_explain")
        obs = summarize_error_events(events)
        if obs["errors_recorded"]:
            print(f"\n  Observability: recorded {obs['errors_recorded']} issue(s): {', '.join(obs['codes'])}")
    except Exception:
        pass

    if report.get("status") != "success":
        print(f"\n  ERROR: {report.get('error')}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Explainability generation failed.                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1

    print(f"\n  Claims input:       {report['claims_input']:,}")
    print(f"  Claims explained:   {report['claims_explained']:,}")
    print(f"  Reason rows:        {report['reason_rows']:,}")
    print(f"  Failed claims:      {report['failed_claim_count']:,}")

    print("\n  Risk-level counts:")
    for level, count in report.get("risk_level_counts", {}).items():
        print(f"    {str(level):<10} {count}")

    print("\n  Top reasons:")
    for reason, count in report.get("top_reason_counts", {}).items():
        print(f"    {reason:<35} {count}")

    print("\n  Saved artifacts:")
    print(f"    Explanations: {report['explanations_path']}")
    print(f"    Summary:      {report['summary_path']}")
    print(f"    Report:       {report['report_path']}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ✓  Week 5 explanations complete. Ready for Week 6 RAG.   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
