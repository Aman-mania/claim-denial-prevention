"""
Dashboard Tab 2 — Clean Data Analysis (Silver Layer)
Shows Silver data quality, cleaning impact, and business logic violations.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics.aggregations import (
    compute_billed_distribution,
    compute_claim_completeness,
    compute_claims_by_diagnosis,
    compute_cleaning_impact,
    compute_cost_analysis,
    compute_null_profile,
    compute_provider_risk_summary,
    compute_specialty_summary,
    compute_violation_summary,
)
from dev_dashboard.components.charts import (
    bar_chart, billed_histogram, completeness_chart,
    cost_comparison_chart, missing_flags_chart,
    null_comparison_chart, null_rate_chart,
    provider_violation_chart, violation_chart,
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_silver(silver_dir: Path) -> tuple[pd.DataFrame, ...]:
    def _read(name: str) -> pd.DataFrame:
        path = silver_dir / name / f"{name}_silver.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _read("claims"), _read("providers"), _read("diagnosis"), _read("cost")


@st.cache_data(ttl=300, show_spinner=False)
def _load_bronze_claims(bronze_dir: Path) -> pd.DataFrame:
    path = bronze_dir / "claims" / "claims_bronze.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _metric_row(metrics: list[tuple]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def render_clean_tab(bronze_dir: Path, silver_dir: Path) -> None:
    st.header("Silver Layer — Cleaned Data Analysis")
    st.caption("Cleaning impact, business logic violations, and data quality after transformation.")

    if not (silver_dir / "claims" / "claims_silver.parquet").exists():
        st.warning("Silver data not found. Run the cleaning pipeline first:", icon="⚠️")
        st.code("python run_silver.py", language="bash")
        return

    with st.spinner("Loading Silver data..."):
        silver_claims, silver_providers, silver_diagnosis, silver_cost = _load_silver(silver_dir)
        bronze_claims = _load_bronze_claims(bronze_dir)

    # ── Section 1: Cleaning impact ─────────────────────────────────────────────
    st.subheader("1 — Cleaning Impact: Bronze → Silver")
    st.caption(
        "Null string codes are filled with **MISSING** sentinel (traceable via flag columns). "
        "Billed amount is never filled."
    )

    impact = compute_cleaning_impact(bronze_claims, silver_claims)
    _metric_row([
        ("Bronze Rows",       str(impact["bronze_rows"]),  None),
        ("Silver Rows",       str(impact["silver_rows"]),
         f"−{impact['rows_removed']} dupes removed" if impact["rows_removed"] else "No dupes"),
        ("Diagnosis Fills",   str(impact["flagged_diag"]),
         f"{impact['flagged_diag']/max(impact['silver_rows'],1)*100:.1f}% had MISSING code"),
        ("Procedure Fills",   str(impact["flagged_proc"]),
         f"{impact['flagged_proc']/max(impact['silver_rows'],1)*100:.1f}% had MISSING code"),
    ])

    st.plotly_chart(null_comparison_chart(impact), key="s_null_comparison",
                    use_container_width=True)
    st.info(
        "**diagnosis_code and procedure_code show 0% null in Silver** because nulls are "
        "replaced with `MISSING` sentinel. The boolean flags `*_missing` preserve the original "
        "null state for ML. **billed_amount stays null** — financial data is never imputed."
    )

    st.divider()

    # ── Section 2: Claim completeness ─────────────────────────────────────────
    st.subheader("2 — Claim Completeness")
    st.caption("How many of the 3 critical fields are present per claim.")

    comp_df = compute_claim_completeness(silver_claims)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(completeness_chart(comp_df), key="s_completeness",
                        use_container_width=True)
    with c2:
        if not comp_df.empty:
            st.markdown("**Breakdown**")
            for _, row in comp_df.iterrows():
                st.markdown(f"- **{row['completeness_level']}**: {row['claim_count']} ({row['pct']}%)")
            st.caption(
                "Complete claims (0 missing) are ready for ML. "
                "Shell claims (all 3 null) should be excluded from training."
            )

    st.divider()

    # ── Section 3: Business logic violations ──────────────────────────────────
    st.subheader("3 — Business Logic Violations")
    st.caption(
        "These are structural issues beyond simple null values — "
        "they indicate claims that **will likely be denied** for specific reasons."
    )

    violations_df = compute_violation_summary(silver_claims)

    # Highlight the two critical ones
    pnd = impact.get("proc_no_diag", 0)
    dnp = impact.get("diag_no_proc", 0)
    total = max(len(silver_claims), 1)
    _metric_row([
        ("Procedure w/o Diagnosis", str(pnd),
         f"{pnd/total*100:.1f}% — missing clinical justification"),
        ("Diagnosis w/o Procedure", str(dnp),
         f"{dnp/total*100:.1f}% — condition documented, nothing billed"),
        ("Amount w/o Any Code", str(impact.get("flagged_amount", 0) - pnd if pnd else impact.get("flagged_amount",0)),
         "Cost recorded but no clinical context"),
        ("Shell Claims (all null)", str((silver_claims[["diagnosis_code_missing","procedure_code_missing","billed_amount_missing"]].sum(axis=1)==3).sum()) if all(c in silver_claims.columns for c in ["diagnosis_code_missing","procedure_code_missing","billed_amount_missing"]) else "N/A", None),
    ])

    st.plotly_chart(violation_chart(violations_df), key="s_violations",
                    use_container_width=True)

    st.caption(
        "🔴 **High severity** = direct denial trigger per payer rules. "
        "🟡 **Medium** = requires review but may pass with supporting documentation. "
        "These will become Rule Engine inputs in Week 7."
    )

    st.divider()

    # ── Section 4: Provider violation risk ────────────────────────────────────
    st.subheader("4 — Provider Violation Risk")
    st.caption("Providers with the highest proportion of flagged claims. Week 5 ML will use this as a risk feature.")

    risk_df = compute_provider_risk_summary(silver_claims, silver_providers)
    if not risk_df.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(provider_violation_chart(risk_df, top_n=15),
                            key="s_provider_violations", use_container_width=True)
        with c2:
            st.markdown("**Top 10 by violation rate**")
            display = risk_df[["provider_id", "specialty", "total_claims",
                                "violation_rate_pct"]].head(10)
            st.dataframe(display, key="s_provider_risk_table",
                         use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 5: Silver claim distributions ─────────────────────────────────
    st.subheader("5 — Silver Claim Distributions")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            bar_chart(compute_claims_by_diagnosis(silver_claims, silver_diagnosis),
                      "diagnosis_code", "claim_count",
                      "Claims by Diagnosis (Silver)", color="success"),
            key="s_by_diagnosis", use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            bar_chart(compute_specialty_summary(silver_claims, silver_providers),
                      "specialty", "claim_count",
                      "Claims by Specialty (Silver)", color="success", horizontal=False),
            key="s_by_specialty", use_container_width=True,
        )

    st.divider()

    # ── Section 6: Billed amount distribution ─────────────────────────────────
    st.subheader("6 — Billed Amount Distribution")
    st.caption(
        "Distribution of billed amounts across the 657 claims with an amount present. "
        "The spread is wide (₹633–₹49,869) with a roughly uniform distribution."
    )
    dist_df = compute_billed_distribution(silver_claims)
    st.plotly_chart(billed_histogram(dist_df), key="s_billed_hist", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    ba = silver_claims["billed_amount"].dropna()
    if not ba.empty:
        c1.metric("Median Billed",    f"₹{ba.median():,.0f}")
        c2.metric("Mean Billed",      f"₹{ba.mean():,.0f}")
        c3.metric("Min Billed",       f"₹{ba.min():,.0f}")
        c4.metric("Max Billed",       f"₹{ba.max():,.0f}")

    st.divider()

    # ── Section 7: Cost analysis ───────────────────────────────────────────────
    st.subheader("7 — Cost Benchmark Analysis")
    st.caption(
        "Billed amounts vs procedure-level expected costs. "
        "⚠️ The cost table maps **one region per procedure** — "
        "regional benchmarking only applies to 96/759 claims (12.6%). "
        "Gold layer (Week 4) will use **procedure-level cost only** for 100% coverage."
    )
    cost_df = compute_cost_analysis(silver_claims, silver_cost)
    if not cost_df.empty:
        st.plotly_chart(cost_comparison_chart(cost_df), key="s_cost_comparison",
                        use_container_width=True)
        med_dev = cost_df["deviation_pct"].median()
        if pd.notna(med_dev):
            st.warning(
                f"Median deviation from expected: **+{med_dev:.0f}%**. "
                f"PROC6 is worst at **+1,738%** above expected. "
                f"These deviations will be the primary ML anomaly signal in Week 5."
            )

    st.divider()

    # ── Section 8: New Silver columns ─────────────────────────────────────────
    with st.expander("📋 New Columns Added by Silver Cleaning"):
        bronze_cols = set(bronze_claims.columns) - {"ingestion_timestamp", "source_file"}
        silver_meta = {"ingestion_timestamp", "source_file", "silver_timestamp"}
        silver_data_cols = set(silver_claims.columns) - silver_meta
        new_cols = sorted(silver_data_cols - bronze_cols)

        col_desc = {
            "diagnosis_code_missing":  "bool — True where diagnosis_code was null in Bronze",
            "procedure_code_missing":  "bool — True where procedure_code was null in Bronze",
            "billed_amount_missing":   "bool — True where billed_amount is null (never filled)",
            "proc_no_diag":            "bool — Procedure present but diagnosis absent",
            "diag_no_proc":            "bool — Diagnosis present but procedure absent",
            "location_missing":        "bool — True where provider location was null",
            "cost_ratio":              "float — average_cost / expected_cost benchmark ratio",
        }

        rows = []
        for col in new_cols:
            dtype = str(silver_claims[col].dtype) if col in silver_claims.columns else "—"
            desc  = col_desc.get(col, "—")
            n_true = int(silver_claims[col].sum()) if col in silver_claims.columns and silver_claims[col].dtype == bool else "—"
            rows.append({"column": col, "dtype": dtype, "description": desc,
                         "true_count": n_true})
        st.dataframe(pd.DataFrame(rows), key="s_new_cols_table",
                     use_container_width=True, hide_index=True)

    with st.expander("📄 Silver Claims Sample (first 50 rows — scroll right for flag columns)"):
        skip = {"ingestion_timestamp", "source_file", "silver_timestamp"}
        preview_cols = [c for c in silver_claims.columns if c not in skip]
        st.dataframe(silver_claims[preview_cols].head(50), key="s_preview_table",
                     use_container_width=True, hide_index=True)
