"""
Dashboard Tab — Week 5 Explainable AI
=====================================
Separate from the Week 4 ML tab.

Shows:
- Explanation artifact status
- Claim-level business reasons
- SHAP contribution details
- Week 6-ready policy queries and tags
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import DASHBOARD_CACHE_TTL


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_explanation_summary(gold_dir: Path) -> pd.DataFrame:
    path = gold_dir / "gold_claim_explanation_summary.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_explanations(gold_dir: Path) -> pd.DataFrame:
    path = gold_dir / "gold_claim_explanations.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_report(gold_dir: Path) -> dict:
    path = gold_dir / "explanation_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []


def _risk_badge(level: str) -> str:
    level = str(level).upper()
    if level == "HIGH":
        return "⛔ HIGH"
    if level == "MEDIUM":
        return "⚠️ MEDIUM"
    return "✅ LOW"


def render_explainability_tab(gold_dir: Path, models_dir: Path) -> None:
    st.header("Week 5 — Explainable AI")
    st.caption(
        "This tab converts model predictions into business reasons and prepares "
        "reason-level policy queries for Week 6 RAG."
    )

    summary = _load_explanation_summary(gold_dir)
    reasons = _load_explanations(gold_dir)
    report = _load_report(gold_dir)

    if summary.empty or reasons.empty:
        st.warning("Explainability artifacts not found. Generate them first:", icon="⚠️")
        st.code("python run_gold.py\npython run_train.py\npython run_explain.py", language="bash")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims explained", f"{summary['claim_id'].nunique():,}")
    c2.metric("Reason rows", f"{len(reasons):,}")
    c3.metric("High risk", f"{(summary['risk_level'].astype(str).str.upper() == 'HIGH').sum():,}")
    c4.metric("Medium risk", f"{(summary['risk_level'].astype(str).str.upper() == 'MEDIUM').sum():,}")

    if report:
        st.caption(
            f"Generated: {report.get('created_at', '—')} · "
            f"Version: {report.get('explanation_version', '—')}"
        )

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        selected_levels = st.multiselect(
            "Risk levels",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"],
        )
        filtered_summary = summary[
            summary["risk_level"].astype(str).str.upper().isin(selected_levels)
        ] if selected_levels else summary

        if filtered_summary.empty:
            st.info("No claims match the selected risk filters.")
            return

        claim_labels = {
            f"{row.claim_id} | {_risk_badge(row.risk_level)} | {row.risk_score:.1%}": row.claim_id
            for row in filtered_summary.itertuples()
        }
        selected_label = st.selectbox("Select claim", list(claim_labels.keys()))
        selected_claim_id = claim_labels[selected_label]

    selected_summary = summary[summary["claim_id"] == selected_claim_id].iloc[0].to_dict()
    selected_reasons = reasons[reasons["claim_id"] == selected_claim_id].sort_values("reason_rank")

    with right:
        st.subheader(f"Claim {selected_claim_id}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{float(selected_summary['risk_score']):.1%}")
        c2.metric("Risk level", str(selected_summary["risk_level"]))
        c3.metric("Prediction", "Likely denied" if int(selected_summary["predicted_denial"]) else "Review / likely ok")

        st.caption(
            f"Review threshold: {float(selected_summary['review_threshold']):.2%} · "
            f"Denial threshold: {float(selected_summary['classification_threshold']):.2%} · "
            f"Model: {selected_summary.get('model_used', '—')}"
        )

    st.markdown("### Business reasons")

    for row in selected_reasons.itertuples():
        with st.expander(f"{row.reason_rank}. {row.reason_title}", expanded=True):
            st.write(row.reason_text)
            st.markdown(f"**Suggested fix:** {row.fix_suggestion}")
            st.markdown(
                f"**Model evidence:** `{row.feature_label}` contributed "
                f"`{float(row.shap_value):+.4f}` raw log-odds "
                f"({row.shap_direction.replace('_', ' ')})."
            )
            tags = _json_list(row.policy_tags)
            if tags:
                st.markdown("**Week 6 policy tags:** " + ", ".join(f"`{t}`" for t in tags))
            st.markdown("**Week 6 retrieval query:**")
            st.code(row.policy_query, language="text")

    st.divider()
    st.markdown("### Explanation table")
    display_cols = [
        "reason_rank",
        "reason_code",
        "reason_title",
        "business_category",
        "feature_label",
        "feature_value",
        "shap_value",
        "fix_suggestion",
    ]
    st.dataframe(
        selected_reasons[[c for c in display_cols if c in selected_reasons.columns]],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Raw Week 6 handoff fields"):
        handoff_cols = ["claim_id", "reason_code", "policy_query", "policy_tags", "reason_text"]
        st.dataframe(
            selected_reasons[[c for c in handoff_cols if c in selected_reasons.columns]],
            use_container_width=True,
            hide_index=True,
        )
