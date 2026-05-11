"""
Risk Explanations dashboard tab.

This tab presents Week 5 explainability artifacts in a demo-friendly way:
- no visible week labels
- no raw/debug-first layout
- stable renderer contract for dev_dashboard/app.py
- unique Streamlit widget keys to avoid duplicate element IDs
- full-width, aligned claim summary + business reason cards
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


EXPLANATIONS_FILE = "gold_claim_explanations.parquet"
SUMMARY_FILE = "gold_claim_explanation_summary.parquet"
REPORT_FILE = "explanation_report.json"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _resolve_gold_dir(gold_dir: str | Path | None = None, root_dir: str | Path | None = None) -> Path:
    if gold_dir is not None:
        return Path(gold_dir)
    if root_dir is not None:
        return Path(root_dir) / "data" / "gold"
    return Path.cwd() / "data" / "gold"


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as exc:  # pragma: no cover - shown in Streamlit UI
        st.warning(f"Could not read {p.name}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_artifacts(gold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    explanations = _read_parquet_cached(str(gold_dir / EXPLANATIONS_FILE))
    summary = _read_parquet_cached(str(gold_dir / SUMMARY_FILE))
    report = _read_json_cached(str(gold_dir / REPORT_FILE))
    return explanations, summary, report


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%" if value <= 1.0 else f"{value:.2f}%"


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _clean_text(value: Any, fallback: str = "—") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _sentence_case(value: Any) -> str:
    text = _clean_text(value, "")
    if not text:
        return "—"
    return text[:1].upper() + text[1:]


def _parse_tags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _risk_sort_key(level: Any) -> int:
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return order.get(str(level).upper(), 99)


def _selected_claim_rows(explanations: pd.DataFrame, claim_id: str) -> pd.DataFrame:
    if explanations.empty or "claim_id" not in explanations.columns:
        return pd.DataFrame()
    rows = explanations[explanations["claim_id"].astype(str) == str(claim_id)].copy()
    if "reason_rank" in rows.columns:
        rows = rows.sort_values("reason_rank", kind="stable")
    elif "shap_value" in rows.columns:
        rows = rows.assign(_abs_shap=rows["shap_value"].abs()).sort_values("_abs_shap", ascending=False)
    return rows


def _claim_summary_from_rows(rows: pd.DataFrame, summary: pd.DataFrame, claim_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"claim_id": claim_id}

    if not summary.empty and "claim_id" in summary.columns:
        s = summary[summary["claim_id"].astype(str) == str(claim_id)]
        if not s.empty:
            result.update(s.iloc[0].to_dict())

    if not rows.empty:
        first = rows.iloc[0].to_dict()
        for key in [
            "risk_score",
            "risk_level",
            "predicted_denial",
            "classification_threshold",
            "review_threshold",
            "model_used",
        ]:
            if key not in result or pd.isna(result.get(key)):
                result[key] = first.get(key)
    return result


def _all_claim_options(explanations: pd.DataFrame, summary: pd.DataFrame, selected_levels: Iterable[str]) -> list[tuple[str, str]]:
    source = summary if not summary.empty and "claim_id" in summary.columns else explanations
    if source.empty or "claim_id" not in source.columns:
        return []

    df = source.copy()
    if "risk_level" in df.columns and selected_levels:
        levels = {str(x).upper() for x in selected_levels}
        df = df[df["risk_level"].astype(str).str.upper().isin(levels)]

    if df.empty:
        return []

    cols = [c for c in ["claim_id", "risk_level", "risk_score"] if c in df.columns]
    df = df[cols].drop_duplicates(subset=["claim_id"])
    if "risk_level" in df.columns:
        df["_risk_sort"] = df["risk_level"].map(_risk_sort_key)
    else:
        df["_risk_sort"] = 99
    if "risk_score" in df.columns:
        df["_score_sort"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    else:
        df["_score_sort"] = 0.0
    df = df.sort_values(["_risk_sort", "_score_sort", "claim_id"], ascending=[True, False, True])

    options: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        claim_id = str(row.get("claim_id"))
        risk = _clean_text(row.get("risk_level"), "UNKNOWN").upper()
        score = _format_pct(row.get("risk_score")) if "risk_score" in row else "—"
        options.append((claim_id, f"{claim_id} | {risk} risk | {score}"))
    return options


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .risk-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 0.65rem;
            padding: 1.0rem 1.1rem;
            background: rgba(250, 250, 252, 0.65);
            margin-bottom: 0.75rem;
        }
        .risk-card-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 0.45rem;
        }
        .risk-card-body {
            font-size: 0.97rem;
            line-height: 1.55;
            margin-bottom: 0.60rem;
        }
        .risk-meta {
            color: #6b7280;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .kpi-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 0.65rem;
            padding: 0.85rem 1rem;
            background: white;
            min-height: 5.9rem;
        }
        .kpi-label {
            color: #6b7280;
            font-size: 0.82rem;
            margin-bottom: 0.28rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 650;
            letter-spacing: 0.01rem;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class=\"kpi-card\">
            <div class=\"kpi-label\">{label}</div>
            <div class=\"kpi-value\">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status(explanations: pd.DataFrame, summary: pd.DataFrame, report: dict[str, Any]) -> None:
    claim_count = 0
    if not summary.empty and "claim_id" in summary.columns:
        claim_count = int(summary["claim_id"].nunique())
    elif not explanations.empty and "claim_id" in explanations.columns:
        claim_count = int(explanations["claim_id"].nunique())

    reason_rows = len(explanations)
    high_count = 0
    medium_count = 0
    if not summary.empty and "risk_level" in summary.columns:
        levels = summary["risk_level"].astype(str).str.upper()
        high_count = int((levels == "HIGH").sum())
        medium_count = int((levels == "MEDIUM").sum())
    elif not explanations.empty and "risk_level" in explanations.columns:
        claim_levels = explanations[["claim_id", "risk_level"]].drop_duplicates("claim_id")
        levels = claim_levels["risk_level"].astype(str).str.upper()
        high_count = int((levels == "HIGH").sum())
        medium_count = int((levels == "MEDIUM").sum())

    cols = st.columns(4)
    cols[0].metric("Claims explained", f"{claim_count:,}", border=True)
    cols[1].metric("Reason rows", f"{reason_rows:,}", border=True)
    cols[2].metric("High risk", f"{high_count:,}", border=True)
    cols[3].metric("Medium risk", f"{medium_count:,}", border=True)

    status = report.get("status") or report.get("run_status")
    if status:
        st.caption(f"Latest explanation run status: {status}")


def _render_claim_header(summary: dict[str, Any]) -> None:
    claim_id = _clean_text(summary.get("claim_id"))
    risk_score = _format_pct(summary.get("risk_score"))
    risk_level = _clean_text(summary.get("risk_level"), "UNKNOWN").upper()
    pred = str(summary.get("predicted_denial", "")).strip()
    if pred in {"1", "1.0", "True", "true"}:
        prediction = "Likely denial"
    elif risk_level == "MEDIUM":
        prediction = "Review recommended"
    elif pred in {"0", "0.0", "False", "false"}:
        prediction = "Likely okay"
    else:
        prediction = "—"

    st.subheader(f"Claim {claim_id}")
    c1, c2, c3 = st.columns(3)
    with c1:
        _metric_card("Risk score", risk_score)
    with c2:
        _metric_card("Risk level", risk_level)
    with c3:
        _metric_card("Prediction", prediction)

    threshold_text = []
    if summary.get("review_threshold") is not None:
        threshold_text.append(f"Review threshold: {_format_pct(summary.get('review_threshold'))}")
    if summary.get("classification_threshold") is not None:
        threshold_text.append(f"Denial threshold: {_format_pct(summary.get('classification_threshold'))}")
    if summary.get("model_used") is not None:
        threshold_text.append(f"Model: {_clean_text(summary.get('model_used'))}")
    if threshold_text:
        st.caption(" · ".join(threshold_text))


def _render_reason_card(row: pd.Series, rank: int) -> None:
    title = _sentence_case(row.get("reason_title") or row.get("reason_code") or f"Reason {rank}")
    text = _sentence_case(row.get("reason_text") or row.get("business_reason") or "No explanation text available.")
    fix = _sentence_case(row.get("fix_suggestion") or row.get("suggested_action") or "Review this claim before submission.")
    feature = _clean_text(row.get("feature_name"), "—")
    feature_value = _clean_text(row.get("feature_value"), "—")
    shap_value = _format_number(row.get("shap_value"), 4)
    direction = _clean_text(row.get("shap_direction"), "—").replace("_", " ")

    tags = _parse_tags(row.get("policy_tags"))
    policy_query = _clean_text(row.get("policy_query"), "")

    with st.container(border=True):
        st.markdown(f"**{rank}. {title}**")
        st.write(text)
        st.markdown(f"**Suggested action:** {fix}")
        st.caption(
            f"Evidence feature: `{feature}` · Value: `{feature_value}` · "
            f"SHAP contribution: `{shap_value}` · Direction: `{direction}`"
        )
        if tags or policy_query:
            with st.expander("Policy retrieval handoff", expanded=False):
                if tags:
                    st.caption("Tags: " + ", ".join(f"`{tag}`" for tag in tags))
                if policy_query:
                    st.code(policy_query, language="text")


def _render_reason_cards(rows: pd.DataFrame) -> None:
    st.subheader("Business reasons")
    if rows.empty:
        st.info("No business reasons are available for the selected claim.")
        return
    display = rows.head(5).reset_index(drop=True)
    for idx, (_, row) in enumerate(display.iterrows(), start=1):
        _render_reason_card(row, idx)


def _render_reason_table(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    columns = [
        "reason_rank",
        "reason_code",
        "reason_title",
        "reason_text",
        "feature_name",
        "feature_value",
        "shap_value",
        "shap_direction",
        "fix_suggestion",
        "policy_tags",
        "policy_query",
    ]
    existing = [c for c in columns if c in rows.columns]
    with st.expander("Reason table", expanded=False):
        st.dataframe(rows[existing], width="stretch", hide_index=True)


def _render_diagnostics(report: dict[str, Any], explanations: pd.DataFrame, summary: pd.DataFrame) -> None:
    with st.expander("Explanation diagnostics", expanded=False):
        diag = {
            "explanation_rows": len(explanations),
            "summary_rows": len(summary),
            "claims_in_explanations": int(explanations["claim_id"].nunique()) if "claim_id" in explanations.columns else 0,
        }
        if report:
            diag.update({k: v for k, v in report.items() if k in {"status", "failed_claim_count", "created_at", "model_used"}})
        st.json(diag)


# ---------------------------------------------------------------------------
# Public renderer contract used by dev_dashboard/app.py
# ---------------------------------------------------------------------------


def render_explainability_tab(
    root_dir: str | Path | None = None,
    gold_dir: str | Path | None = None,
    models_dir: str | Path | None = None,  # kept for app.py compatibility
    **_: Any,
) -> None:
    """Render the Risk Explanations dashboard tab."""
    _inject_styles()

    resolved_gold_dir = _resolve_gold_dir(gold_dir=gold_dir, root_dir=root_dir)
    explanations, summary, report = _load_artifacts(resolved_gold_dir)

    st.header("Risk Explanations")
    st.caption("Model risk signals translated into business-readable claim review reasons.")

    if explanations.empty:
        st.warning(
            "Explanation artifacts are missing. Run `python run_explain.py` after training the model."
        )
        return

    _render_status(explanations, summary, report)
    st.divider()

    risk_levels = sorted(
        [str(x).upper() for x in explanations.get("risk_level", pd.Series(dtype=str)).dropna().unique()],
        key=_risk_sort_key,
    )
    if not risk_levels:
        risk_levels = ["HIGH", "MEDIUM", "LOW"]

    filter_col, select_col = st.columns([1.0, 2.2], vertical_alignment="bottom")
    with filter_col:
        selected_levels = st.multiselect(
            "Risk levels",
            options=risk_levels,
            default=[lvl for lvl in ["HIGH", "MEDIUM"] if lvl in risk_levels] or risk_levels[:2],
            key="risk_explanations_level_filter",
        )
    options = _all_claim_options(explanations, summary, selected_levels)
    if not options:
        st.info("No claims match the selected filters.")
        return

    option_labels = [label for _, label in options]
    with select_col:
        selected_label = st.selectbox(
            "Select claim",
            option_labels,
            index=0,
            key="risk_explanations_claim_select",
        )
    selected_claim = options[option_labels.index(selected_label)][0]

    rows = _selected_claim_rows(explanations, selected_claim)
    claim_summary = _claim_summary_from_rows(rows, summary, selected_claim)

    st.divider()
    _render_claim_header(claim_summary)
    st.write("")
    _render_reason_cards(rows)
    _render_reason_table(rows)
    st.divider()
    _render_diagnostics(report, explanations, summary)


# Backward-compatible aliases. Do not remove: older local patches/scripts may import these.
render_risk_explanations_tab = render_explainability_tab
render_xai_tab = render_explainability_tab
