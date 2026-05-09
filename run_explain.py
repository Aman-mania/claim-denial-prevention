#!/usr/bin/env python3
"""
Week 5 — Explainable AI Entry Point
===================================
Gold features + trained XGBoost model + SHAP → business reasons table.

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
from src.observability import summarize_error_events, tracker_from_env

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
    tracker = tracker_from_env()

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
        error_tracker=tracker,
        model_name="xgboost",
    )
    report = pipeline.run(limit=args.limit, claim_ids=args.claim_ids)

    obs = summarize_error_events([])
    try:
        repeated = tracker.get_repeated_errors(min_count=2)
        obs["repeated_errors"] = len(repeated)
    except Exception:
        pass

    if report.get("status") not in {"success", "success_with_warnings"}:
        print(f"\n  ERROR: {report.get('error')}")
        if report.get("error_code"):
            print(f"  Error code: {report.get('error_code')}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Explainability generation failed.                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1

    print(f"\n  Status:             {report['status']}")
    print(f"  Claims input:       {report['claims_input']:,}")
    print(f"  Claims explained:   {report['claims_explained']:,}")
    print(f"  Reason rows:        {report['reason_rows']:,}")
    print(f"  Failed claims:      {report['failed_claim_count']:,}")
    if obs.get("repeated_errors"):
        print(f"  Repeated errors:    {obs['repeated_errors']:,}  (see logs/error_summary.json)")

    print("\n  Risk-level counts:")
    for level, count in report.get("risk_level_counts", {}).items():
        print(f"    {str(level):<10} {count}")

    print("\n  Top reasons:")
    for reason, count in report.get("top_reason_counts", {}).items():
        print(f"    {reason:<35} {count}")

    if report.get("unmapped_features"):
        print("\n  Unmapped SHAP features seen:")
        for feature, count in sorted(report["unmapped_features"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {feature:<35} {count}")

    print("\n  Saved artifacts:")
    print(f"    Explanations: {report['explanations_path']}")
    print(f"    Summary:      {report['summary_path']}")
    print(f"    Report:       {report['report_path']}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    if report["status"] == "success_with_warnings":
        print("║   ⚠  Week 5 completed with warnings. Check error logs.      ║")
    else:
        print("║   ✓  Week 5 explanations complete. Ready for Week 6 RAG.   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
