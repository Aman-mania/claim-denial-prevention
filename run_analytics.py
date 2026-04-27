#!/usr/bin/env python3
"""
Week 2 — Analytics Pipeline Entry Point
==========================================
Loads Bronze data, computes all analytics aggregations,
and prints a summary report to stdout.

Usage:
    python run_analytics.py

Requires Bronze layer (run_ingestion.py) to exist first.
Outputs a human-readable summary — the full analytics are
available interactively via the dev dashboard.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.analytics.aggregations import (
    compute_claims_by_diagnosis,
    compute_claims_by_provider,
    compute_cost_analysis,
    compute_high_cost_claims,
    compute_null_profile,
    compute_overview,
    compute_specialty_summary,
)

import pandas as pd

BASE_DIR   = Path(__file__).parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def _load(name: str) -> pd.DataFrame:
    path = BRONZE_DIR / name / f"{name}_bronze.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Bronze file missing: {path}. Run run_ingestion.py first.")
    return pd.read_parquet(path)


def main() -> int:
    setup_logging(level="INFO")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 2: Analytics Summary       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        claims    = _load("claims")
        providers = _load("providers")
        diagnosis = _load("diagnosis")
        cost      = _load("cost")
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        return 1

    # Overview
    overview = compute_overview(claims, providers, diagnosis, cost)
    print(f"\n  Total claims:       {overview['total_claims']:,}")
    print(f"  Unique patients:    {overview['unique_patients']:,}")
    print(f"  Unique providers:   {overview['unique_providers']:,}")
    print(f"  Date range:         {overview['date_min']} → {overview['date_max']}")
    print(f"  Avg billed amount:  ₹{overview['avg_billed_amount']:,.0f}")
    print(f"  Complete claims:    {overview['claims_complete']:,}")
    print(f"  Shell claims:       {overview['shell_claims']:,}")

    # Null profile
    null_df = compute_null_profile(claims, "claims")
    print("\n  Null rates (claims):")
    for _, row in null_df[null_df["null_pct"] > 0].iterrows():
        print(f"    {row['column']:<24} {row['null_pct']:>5.1f}%")

    # Top providers
    by_provider = compute_claims_by_provider(claims, providers)
    print("\n  Top 5 providers by claim count:")
    for _, row in by_provider.head(5).iterrows():
        print(f"    {row['provider_id']}  →  {row['claim_count']} claims  "
              f"(avg ₹{row['avg_billed']:,.0f})")

    # Diagnosis breakdown
    by_diag = compute_claims_by_diagnosis(claims, diagnosis)
    print("\n  Claims by diagnosis:")
    for _, row in by_diag.head(6).iterrows():
        code = row["diagnosis_code"] if pd.notna(row["diagnosis_code"]) else "MISSING"
        print(f"    {code:<8}  {row['claim_count']:>5} claims")

    # Cost analysis
    cost_df = compute_cost_analysis(claims, cost)
    if not cost_df.empty:
        print("\n  Cost deviation by procedure (avg billed vs expected):")
        for _, row in cost_df.dropna(subset=["deviation_pct"]).iterrows():
            print(f"    {row['procedure_code']:<8}  "
                  f"avg ₹{row['avg_billed']:>8,.0f}  "
                  f"expected ₹{row['expected_cost']:>6,.0f}  "
                  f"deviation {row['deviation_pct']:>+.0f}%")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Analytics complete. Launch dashboard:                     ║")
    print("║   streamlit run dev_dashboard/app.py                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
