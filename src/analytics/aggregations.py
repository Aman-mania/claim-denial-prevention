"""
Analytics Layer — Aggregation Functions
=========================================
Pure transformation functions: DataFrame in → analytics DataFrame/dict out.
No Streamlit, no file I/O, no side effects.

Consumed by:
  - run_analytics.py   (saves results to data/analytics/)
  - dev_dashboard/tabs/ (reads Parquet or calls these directly with @st.cache_data)
  - tests/analytics/   (unit tested independently)

Each function is independently callable and testable.
Handles nulls/empty DataFrames without crashing.
"""

from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Columns added by ingestion/silver pipelines — excluded from analysis
_META_COLS = frozenset({"ingestion_timestamp", "source_file", "silver_timestamp"})


def _data_cols(df: pd.DataFrame) -> list[str]:
    """Return column names excluding pipeline metadata columns."""
    return [c for c in df.columns if c not in _META_COLS]


# ── Overview ──────────────────────────────────────────────────────────────────

def compute_overview(
    claims_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> dict:
    """
    High-level summary statistics for the claims dataset.
    Returns a flat dict of scalar metrics (JSON-serialisable).
    """
    if claims_df.empty:
        logger.warning("compute_overview_empty_claims")
        return {k: 0 for k in [
            "total_claims", "unique_patients", "unique_providers",
            "avg_billed_amount", "total_billed", "claims_complete", "shell_claims",
        ]} | {"date_min": "N/A", "date_max": "N/A"}

    df = claims_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Complete = all three critical fields present
    critical = ["diagnosis_code", "procedure_code", "billed_amount"]
    complete_mask = df[critical].notna().all(axis=1)
    shell_mask    = df[critical].isna().all(axis=1)

    date_min = df["date"].dropna().min()
    date_max = df["date"].dropna().max()

    return {
        "total_claims":      len(df),
        "unique_patients":   int(df["patient_id"].nunique()),
        "unique_providers":  int(df["provider_id"].nunique()),
        "date_min":          str(date_min.date()) if pd.notna(date_min) else "N/A",
        "date_max":          str(date_max.date()) if pd.notna(date_max) else "N/A",
        "avg_billed_amount": round(float(df["billed_amount"].mean()), 2)
                             if df["billed_amount"].notna().any() else 0.0,
        "total_billed":      round(float(df["billed_amount"].sum()), 2),
        "claims_complete":   int(complete_mask.sum()),
        "shell_claims":      int(shell_mask.sum()),
    }


# ── Null profile ──────────────────────────────────────────────────────────────

def compute_null_profile(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Null rate per column for any DataFrame.
    Returns: DataFrame[dataset, column, null_count, null_pct, present_count].
    Works on both Bronze and Silver data.
    """
    cols  = _data_cols(df)
    total = len(df)

    rows = []
    for col in cols:
        nc = int(df[col].isnull().sum())
        rows.append({
            "dataset":       dataset_name,
            "column":        col,
            "null_count":    nc,
            "null_pct":      round(nc / total * 100, 2) if total > 0 else 0.0,
            "present_count": total - nc,
        })

    return pd.DataFrame(rows)


# ── Claims distributions ───────────────────────────────────────────────────────

def compute_claims_by_provider(
    claims_df: pd.DataFrame,
    providers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Claim count and billing stats per provider.
    Left-joins with providers for specialty and location.
    Sorted by claim count descending.
    """
    if claims_df.empty:
        return pd.DataFrame()

    agg = (
        claims_df
        .groupby("provider_id", dropna=False)
        .agg(
            claim_count=("claim_id",       "count"),
            avg_billed= ("billed_amount",  "mean"),
            total_billed=("billed_amount", "sum"),
        )
        .reset_index()
    )

    # Only pull needed columns from providers (guard against missing cols)
    meta_cols = [c for c in ["provider_id", "specialty", "location"] if c in providers_df.columns]
    merged = agg.merge(providers_df[meta_cols], on="provider_id", how="left")

    merged["avg_billed"]   = merged["avg_billed"].round(2)
    merged["total_billed"] = merged["total_billed"].round(2)

    return merged.sort_values("claim_count", ascending=False).reset_index(drop=True)


def compute_claims_by_diagnosis(
    claims_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Claim count per diagnosis code, enriched with category/severity from reference table.
    NULL diagnosis codes are kept as a separate row (they represent missing-code claims).
    """
    if claims_df.empty:
        return pd.DataFrame()

    agg = (
        claims_df
        .groupby("diagnosis_code", dropna=False)
        .agg(
            claim_count=("claim_id",      "count"),
            avg_billed= ("billed_amount", "mean"),
        )
        .reset_index()
    )

    merged = agg.merge(diagnosis_df, on="diagnosis_code", how="left")
    merged["avg_billed"] = merged["avg_billed"].round(2)

    return merged.sort_values("claim_count", ascending=False).reset_index(drop=True)


def compute_specialty_summary(
    claims_df: pd.DataFrame,
    providers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Claim count and billing totals grouped by provider specialty."""
    if claims_df.empty:
        return pd.DataFrame()

    merged = claims_df.merge(
        providers_df[["provider_id", "specialty"]], on="provider_id", how="left"
    )
    return (
        merged
        .groupby("specialty", dropna=False)
        .agg(
            claim_count=  ("claim_id",      "count"),
            avg_billed=   ("billed_amount", "mean"),
            total_billed= ("billed_amount", "sum"),
        )
        .round(2)
        .reset_index()
        .sort_values("claim_count", ascending=False)
        .reset_index(drop=True)
    )


def compute_claims_timeline(claims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily claim count time series.
    Returns: DataFrame[date_str, claim_count] sorted by date ascending.
    """
    if claims_df.empty:
        return pd.DataFrame(columns=["date_str", "claim_count"])

    df = claims_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    timeline = (
        df.dropna(subset=["date"])
        .groupby(df["date"].dt.to_period("D"))
        .agg(claim_count=("claim_id", "count"))
        .reset_index()
    )
    # Convert Period to string for JSON/Parquet compatibility
    timeline["date_str"] = timeline["date"].astype(str)
    return timeline[["date_str", "claim_count"]].sort_values("date_str").reset_index(drop=True)


# ── Cost analysis ─────────────────────────────────────────────────────────────

def compute_cost_analysis(
    claims_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-procedure billing stats vs expected cost benchmarks.
    deviation_pct = (avg_billed - expected_cost) / expected_cost * 100
    Only includes claims where billed_amount is present.
    """
    if claims_df.empty or cost_df.empty:
        return pd.DataFrame()

    df = claims_df[claims_df["billed_amount"].notna()].copy()

    agg = (
        df.groupby("procedure_code", dropna=False)
        .agg(
            claim_count= ("claim_id",      "count"),
            avg_billed=  ("billed_amount", "mean"),
            min_billed=  ("billed_amount", "min"),
            max_billed=  ("billed_amount", "max"),
        )
        .reset_index()
    )

    merged = agg.merge(cost_df, on="procedure_code", how="left")

    # Compute deviation where expected_cost is available
    has_expected = merged["expected_cost"].notna()
    merged.loc[has_expected, "deviation_pct"] = (
        (merged.loc[has_expected, "avg_billed"] - merged.loc[has_expected, "expected_cost"])
        / merged.loc[has_expected, "expected_cost"] * 100
    ).round(2)

    for col in ["avg_billed", "min_billed", "max_billed"]:
        merged[col] = merged[col].round(2)

    return merged.sort_values("claim_count", ascending=False).reset_index(drop=True)


def compute_high_cost_claims(
    claims_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    threshold_pct: float = 100.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Claims where billed_amount exceeds expected_cost by > threshold_pct.
    Returns top_n by deviation descending.
    """
    if claims_df.empty or cost_df.empty:
        return pd.DataFrame()

    merged = claims_df.merge(cost_df[["procedure_code", "expected_cost", "region"]],
                              on="procedure_code", how="inner")
    merged = merged[merged["billed_amount"].notna()].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["deviation_pct"] = (
        (merged["billed_amount"] - merged["expected_cost"]) / merged["expected_cost"] * 100
    ).round(2)

    return (
        merged[merged["deviation_pct"] > threshold_pct]
        .sort_values("deviation_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Silver-specific ────────────────────────────────────────────────────────────

def compute_cleaning_impact(
    bronze_claims: pd.DataFrame,
    silver_claims: pd.DataFrame,
) -> dict:
    """
    Before/after comparison showing the impact of Silver cleaning.
    Returns a flat dict of comparison metrics.
    Used exclusively by the Silver dashboard tab.
    """
    b, s = bronze_claims, silver_claims

    def null_pct(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or df.empty:
            return 0.0
        return round(df[col].isnull().mean() * 100, 2)

    # Count flag columns only if they exist in silver
    flag_cols = ["diagnosis_code_missing", "procedure_code_missing", "billed_amount_missing"]

    return {
        "bronze_rows":               len(b),
        "silver_rows":               len(s),
        "rows_removed":              len(b) - len(s),
        # Null rates in Bronze (raw)
        "bronze_diag_null_pct":      null_pct(b, "diagnosis_code"),
        "bronze_proc_null_pct":      null_pct(b, "procedure_code"),
        "bronze_amount_null_pct":    null_pct(b, "billed_amount"),
        # Null rates in Silver (after cleaning)
        "silver_diag_null_pct":      null_pct(s, "diagnosis_code"),
        "silver_proc_null_pct":      null_pct(s, "procedure_code"),
        "silver_amount_null_pct":    null_pct(s, "billed_amount"),
        # Missing-field flags
        "flagged_diag":   int(s["diagnosis_code_missing"].sum())   if "diagnosis_code_missing"   in s.columns else 0,
        "flagged_proc":   int(s["procedure_code_missing"].sum())   if "procedure_code_missing"   in s.columns else 0,
        "flagged_amount": int(s["billed_amount_missing"].sum())    if "billed_amount_missing"    in s.columns else 0,
        # Business logic violations
        "proc_no_diag":   int(s["proc_no_diag"].sum())            if "proc_no_diag"             in s.columns else 0,
        "diag_no_proc":   int(s["diag_no_proc"].sum())            if "diag_no_proc"             in s.columns else 0,
        # Date parse success rate
        "date_parsed_pct": round(
            pd.to_datetime(s["date"], errors="coerce").notna().mean() * 100, 2
        ) if "date" in s.columns else 0.0,
    }


# ── Silver-specific analytics ──────────────────────────────────────────────────

def compute_claim_completeness(silver_claims: pd.DataFrame) -> pd.DataFrame:
    """
    Breaks claims into completeness buckets based on how many of the 3
    critical fields (diagnosis, procedure, amount) are present.
    Returns: DataFrame[completeness_level, claim_count, pct].
    """
    if silver_claims.empty:
        return pd.DataFrame()

    flag_cols = ["diagnosis_code_missing", "procedure_code_missing", "billed_amount_missing"]
    if not all(c in silver_claims.columns for c in flag_cols):
        return pd.DataFrame()

    # Count how many critical fields are MISSING per claim
    missing_count = silver_claims[flag_cols].sum(axis=1)

    labels = {0: "Complete (0 missing)", 1: "1 field missing",
              2: "2 fields missing",   3: "Shell (all 3 missing)"}
    rows = []
    for n, label in labels.items():
        count = int((missing_count == n).sum())
        rows.append({
            "completeness_level": label,
            "missing_count": n,
            "claim_count": count,
            "pct": round(count / len(silver_claims) * 100, 1),
        })
    return pd.DataFrame(rows)


def compute_violation_summary(silver_claims: pd.DataFrame) -> pd.DataFrame:
    """
    Counts business logic violations detected in Silver.
    Returns a DataFrame suitable for display or charting.
    """
    if silver_claims.empty:
        return pd.DataFrame()

    violations = []
    checks = [
        ("proc_no_diag",          "Procedure w/o Diagnosis",     "High"),
        ("diag_no_proc",          "Diagnosis w/o Procedure",     "Medium"),
        ("billed_amount_missing", "Amount Missing",              "Medium"),
        ("diagnosis_code_missing","Diagnosis Missing",           "High"),
        ("procedure_code_missing","Procedure Missing",           "Medium"),
    ]
    for col, label, severity in checks:
        if col in silver_claims.columns:
            count = int(silver_claims[col].sum())
            violations.append({
                "violation": label,
                "severity": severity,
                "claim_count": count,
                "pct_of_claims": round(count / len(silver_claims) * 100, 1),
            })
    return pd.DataFrame(violations).sort_values("claim_count", ascending=False).reset_index(drop=True)


def compute_provider_risk_summary(
    silver_claims: pd.DataFrame,
    silver_providers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-provider breakdown of violation rates.
    Identifies high-risk providers — useful for the rule engine later.
    """
    if silver_claims.empty or silver_providers.empty:
        return pd.DataFrame()

    violation_cols = [c for c in ["proc_no_diag", "diag_no_proc", "billed_amount_missing",
                                   "diagnosis_code_missing", "procedure_code_missing"]
                      if c in silver_claims.columns]
    if not violation_cols:
        return pd.DataFrame()

    agg = (
        silver_claims
        .groupby("provider_id")
        .agg(
            total_claims       = ("claim_id",       "count"),
            avg_billed         = ("billed_amount",  "mean"),
            **{col: (col, "sum") for col in violation_cols},
        )
        .reset_index()
    )
    agg["total_violations"] = agg[violation_cols].sum(axis=1)
    agg["violation_rate_pct"] = (agg["total_violations"] / agg["total_claims"] * 100).round(1)

    merged = agg.merge(
        silver_providers[["provider_id", "specialty", "location"]],
        on="provider_id", how="left"
    )
    return merged.sort_values("violation_rate_pct", ascending=False).reset_index(drop=True)


def compute_billed_distribution(silver_claims: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """
    Histogram bins for billed_amount distribution.
    Returns: DataFrame[bin_label, claim_count] for charting.
    """
    if silver_claims.empty or "billed_amount" not in silver_claims.columns:
        return pd.DataFrame()

    amounts = silver_claims["billed_amount"].dropna()
    if amounts.empty:
        return pd.DataFrame()

    counts, edges = __import__("numpy").histogram(amounts, bins=n_bins)
    rows = []
    for i, count in enumerate(counts):
        label = f"₹{edges[i]/1000:.0f}k–{edges[i+1]/1000:.0f}k"
        rows.append({"bin_label": label, "bin_start": edges[i], "claim_count": int(count)})
    return pd.DataFrame(rows)
