"""
Dashboard Tab 1 — Raw Data Analysis (Bronze Layer)
Loads Bronze Parquet files and displays unprocessed data analytics.
All computation delegated to src.analytics.aggregations.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import DASHBOARD_CACHE_TTL

from src.analytics.aggregations import (
    compute_claims_by_diagnosis, compute_claims_by_provider,
    compute_claims_timeline, compute_cost_analysis,
    compute_high_cost_claims, compute_null_profile,
    compute_overview, compute_specialty_summary,
)
from dev_dashboard.components.charts import (
    bar_chart, claims_timeline_chart, cost_comparison_chart, null_rate_chart,
)


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_bronze(bronze_dir: Path) -> tuple[pd.DataFrame, ...]:
    def _read(name: str) -> pd.DataFrame:
        path = bronze_dir / name / f"{name}_bronze.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _read("claims"), _read("providers"), _read("diagnosis"), _read("cost")


def _metric_row(metrics: list[tuple]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def render_raw_tab(bronze_dir: Path) -> None:
    st.header("Bronze Layer — Raw Data Analysis")
    st.caption("Unprocessed data as ingested from CSV files. Nulls and issues preserved as-is.")

    with st.spinner("Loading Bronze data..."):
        claims, providers, diagnosis, cost = _load_bronze(bronze_dir)

    if claims.empty:
        st.error("Bronze claims not found. Run `python run_ingestion.py` first.")
        return

    # Overview
    st.subheader("Overview")
    overview = compute_overview(claims, providers, diagnosis, cost)
    _metric_row([
        ("Total Claims",     str(overview["total_claims"]),     None),
        ("Unique Patients",  str(overview["unique_patients"]),  None),
        ("Unique Providers", str(overview["unique_providers"]), None),
        ("Date Range", f"{overview['date_min']} → {overview['date_max']}", None),
    ])
    _metric_row([
        ("Avg Billed",     f"₹{overview['avg_billed_amount']:,.0f}", None),
        ("Total Billed",   f"₹{overview['total_billed']:,.0f}",      None),
        ("Complete Claims", str(overview["claims_complete"]),         None),
        ("Shell Claims (all-null)", str(overview["shell_claims"]),
         f"-{overview['shell_claims']/overview['total_claims']*100:.1f}% of total"
         if overview["total_claims"] > 0 else None),
    ])

    st.divider()

    # Data quality
    st.subheader("Data Quality — Null Rates")
    st.caption("Null rates across all columns in the raw claims dataset.")
    null_df = compute_null_profile(claims, "claims")
    st.plotly_chart(null_rate_chart(null_df, "Claims — Null Rate by Column"),
                    key="raw_null_rate", use_container_width=True)

    for col_label, col in [("Diagnosis Code", "diagnosis_code"),
                            ("Procedure Code", "procedure_code"),
                            ("Billed Amount",  "billed_amount")]:
        row = null_df[null_df["column"] == col]
        if not row.empty:
            pct = row.iloc[0]["null_pct"]
            c = "red" if pct > 25 else "orange" if pct > 10 else "green"
            st.markdown(f":{c}[**{col_label}**: {pct}% null ({row.iloc[0]['null_count']:,} of {len(claims):,} rows)]")

    st.divider()

    # Claims distribution
    st.subheader("Claims Distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(claims_timeline_chart(compute_claims_timeline(claims)),
                        key="raw_timeline", use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(compute_claims_by_diagnosis(claims, diagnosis),
                        "diagnosis_code", "claim_count", "Claims by Diagnosis Code"),
                        key="raw_by_diagnosis", use_container_width=True)

    st.divider()

    # Provider analysis
    st.subheader("Provider Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(compute_claims_by_provider(claims, providers),
                        "provider_id", "claim_count", "Top 15 Providers", top_n=15),
                        key="raw_by_provider", use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(compute_specialty_summary(claims, providers),
                        "specialty", "claim_count", "Claims by Specialty",
                        color="neutral", horizontal=False),
                        key="raw_by_specialty", use_container_width=True)

    st.divider()

    # Cost analysis
    st.subheader("Cost Analysis")
    st.caption("Avg billed vs expected cost benchmarks. Overbilling = primary denial risk.")
    cost_df = compute_cost_analysis(claims, cost)
    if not cost_df.empty:
        st.plotly_chart(cost_comparison_chart(cost_df),
                        key="raw_cost_comparison", use_container_width=True)
        median_dev = cost_df["deviation_pct"].median()
        if pd.notna(median_dev):
            st.info(
                f"**Median overbilling: +{median_dev:.0f}%** above expected cost. "
                f"{(cost_df['deviation_pct'] > 50).sum()} of {len(cost_df)} "
                f"procedure types are billed >50% above benchmark."
            )

    st.divider()

    # Top overbilled
    st.subheader("Top Overbilled Claims")
    high_cost = compute_high_cost_claims(claims, cost, threshold_pct=100.0, top_n=10)
    if not high_cost.empty:
        display_cols = [c for c in ["claim_id", "provider_id", "procedure_code",
                        "billed_amount", "expected_cost", "deviation_pct", "region"]
                        if c in high_cost.columns]
        st.dataframe(high_cost[display_cols], key="raw_high_cost_table",
                     use_container_width=True, hide_index=True)
    else:
        st.info("No claims exceed 100% over expected cost for matched procedures.")

    with st.expander("Raw Claims Sample (first 50 rows)"):
        preview_cols = [c for c in claims.columns if c not in ("ingestion_timestamp", "source_file")]
        st.dataframe(claims[preview_cols].head(50), key="raw_preview_table",
                     use_container_width=True, hide_index=True)
