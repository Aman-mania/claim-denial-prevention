#!/usr/bin/env python3
"""Week 6 — Retrieve policy evidence for Week 5 claim reasons.

Requires:
  python run_explain.py
  python run_policy_ingest.py

Outputs:
  data/gold/gold_claim_policy_matches.parquet
  data/gold/gold_claim_final_explanations.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import setup_logging
from src.observability import ErrorCode, tracker_from_env
from src.rag.policy_matcher import PolicyMatcher
from src.rag.schemas import DEFAULT_MIN_SCORE, DEFAULT_TOP_K

BASE_DIR = Path(__file__).parent
DEFAULT_GOLD_DIR = BASE_DIR / "data" / "gold"
DEFAULT_VECTOR_DIR = BASE_DIR / "data" / "vector_store"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve policy evidence for Week 5 explanations.")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of claims to process for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level="INFO")
    tracker = tracker_from_env()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 6: Reason → Policy Match   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        matcher = PolicyMatcher(
            gold_dir=args.gold_dir,
            vector_dir=args.vector_dir,
            error_tracker=tracker,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        report = matcher.run(limit=args.limit)
        if report["status"] not in {"success", "success_with_warnings"}:
            raise RuntimeError(report.get("error", "policy matching failed"))

        print(f"\n  Reason rows processed:       {report['reason_rows_input']:,}")
        print(f"  Policy match rows:          {report['policy_match_rows']:,}")
        print(f"  Final explanation rows:     {report['final_explanation_rows']:,}")
        print(f"  Unmatched reason count:     {report['unmatched_reason_count']:,}")
        print(f"  Policy matches:             {report['policy_match_path']}")
        print(f"  Final explanations:         {report['final_explanation_path']}")
        print("\n╔══════════════════════════════════════════════════════════════╗")
        if report["status"] == "success_with_warnings":
            print("║   ⚠  Policy matching complete with warnings.                ║")
        else:
            print("║   ✓  Policy matching complete.                              ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 0
    except Exception as exc:
        event = tracker.record_exception(
            exc,
            component="rag",
            stage="policy_match_run_script",
            fallback_code=ErrorCode.RAG_UNEXPECTED,
            metadata={"stage": "policy_match_run_script"},
        )
        print(f"\n  ERROR: {exc}")
        print(f"  Error code: {event.error_code}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Policy matching failed.                               ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1


if __name__ == "__main__":
    sys.exit(main())
