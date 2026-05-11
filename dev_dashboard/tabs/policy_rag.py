"""
Policy Evidence dashboard tab.

This module intentionally keeps a stable public render function signature because
`dev_dashboard/app.py` passes shared project paths into each tab renderer:

    render_policy_rag_tab(root_dir=..., gold_dir=..., models_dir=...)

The UI is presentation-focused: it hides internal week labels, removes excessive
emoji/status noise, de-duplicates repeated policy matches, and shows policy-backed
claim explanations in a clean analyst-facing format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


TEXT_PREVIEW_CHARS = 420


def _as_path(value: Any, fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(value)


def _project_paths(
    root_dir: Path | str | None = None,
    gold_dir: Path | str | None = None,
    **_: Any,
) -> dict[str, Path]:
    root = _as_path(root_dir, Path.cwd())
    gold = _as_path(gold_dir, root / "data" / "gold")
    return {
        "root": root,
        "gold": gold,
        "chunks": root / "data" / "policies" / "processed" / "policy_chunks.parquet",
        "vector_meta": root / "data" / "vector_store" / "policy_metadata.json",
        "vector_npy": root / "data" / "vector_store" / "policy_vectors.npy",
        "vector_faiss": root / "data" / "vector_store" / "policy.faiss",
        "matches": gold / "gold_claim_policy_matches.parquet",
        "finals": gold / "gold_claim_final_explanations.parquet",
        "report": gold / "policy_match_report.json",
    }


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns  # cache invalidation key
    return pd.read_parquet(path_str)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_parquet_cached(str(path), path.stat().st_mtime_ns)
    except Exception as exc:  # noqa: BLE001 - dashboard should not crash on a bad table
        st.error(f"Could not read table: {path}")
        st.caption(str(exc))
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json_cached(str(path), path.stat().st_mtime_ns)
    except Exception:
        return {}


def _status_text(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def _safe_first(row: pd.Series | dict[str, Any], candidates: Iterable[str], default: Any = "") -> Any:
    for name in candidates:
        try:
            value = row.get(name, default)  # type: ignore[union-attr]
        except AttributeError:
            value = default
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and value != "":
            return value
    return default


def _as_percent(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "—"
    if val <= 1:
        val *= 100
    return f"{val:.2f}%"


def _preview_text(value: Any, limit: int = TEXT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _deduplicate_policy_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate retrieval rows while preserving the best score first."""
    if df.empty:
        return df

    result = df.copy()
    score_col = _find_column(result, ["similarity_score", "score", "retrieval_score", "policy_score"])
    if score_col:
        result[score_col] = pd.to_numeric(result[score_col], errors="coerce")
        result = result.sort_values(score_col, ascending=False, na_position="last")

    subset_options = [
        ["claim_id", "reason_code", "policy_chunk_id"],
        ["claim_id", "reason_code", "source_name", "section_title", "policy_text"],
        ["claim_id", "reason_code", "source_name", "policy_text"],
        ["claim_id", "reason_text", "policy_text"],
    ]
    for subset in subset_options:
        existing = [col for col in subset if col in result.columns]
        if len(existing) >= 3:
            return result.drop_duplicates(subset=existing, keep="first")

    return result.drop_duplicates(keep="first")


def _deduplicate_reasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [col for col in ["claim_id", "reason_code", "reason_title", "reason_text"] if col in df.columns]
    if len(subset) >= 2:
        return df.drop_duplicates(subset=subset, keep="first")
    return df.drop_duplicates(keep="first")


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _display_status_cards(paths: dict[str, Path], chunks: pd.DataFrame, matches: pd.DataFrame, finals: pd.DataFrame) -> None:
    cols = st.columns(4)
    cards = [
        ("Policy chunks", paths["chunks"]),
        ("Vector index", paths["vector_meta"]),
        ("Policy matches", paths["matches"]),
        ("Final outputs", paths["finals"]),
    ]
    for col, (label, path) in zip(cols, cards):
        with col:
            st.caption(label)
            st.write(_status_text(path))

    st.divider()
    cols = st.columns(4)
    with cols[0]:
        st.metric("Chunks", f"{len(chunks):,}" if not chunks.empty else "0")
    with cols[1]:
        source_col = _find_column(chunks, ["source_name", "document_id", "source_file"])
        sources = chunks[source_col].nunique() if source_col and not chunks.empty else 0
        st.metric("Policy sources", f"{sources:,}")
    with cols[2]:
        vector_meta = _read_json(paths["vector_meta"])
        dim = vector_meta.get("embedding_dim") or vector_meta.get("dimension") or vector_meta.get("dim") or "—"
        st.metric("Embedding dim", str(dim))
    with cols[3]:
        st.metric("Match rows", f"{len(matches):,}" if not matches.empty else "0")


def _display_claim_summary(finals: pd.DataFrame, matches: pd.DataFrame) -> None:
    claim_source = finals if not finals.empty and "claim_id" in finals.columns else matches
    if claim_source.empty or "claim_id" not in claim_source.columns:
        st.info("No claim-level policy outputs found. Run `python run_week6.py` after generating risk explanations.")
        return

    claim_ids = sorted(str(x) for x in claim_source["claim_id"].dropna().unique())
    if not claim_ids:
        st.info("No claim IDs are available in the policy output tables.")
        return

    selected = st.selectbox("Select claim", claim_ids, index=0, key="policy_evidence_claim_select")
    final_row = pd.Series(dtype="object")
    if not finals.empty and "claim_id" in finals.columns:
        subset = finals[finals["claim_id"].astype(str) == selected]
        if not subset.empty:
            final_row = subset.iloc[0]

    claim_matches = pd.DataFrame()
    if not matches.empty and "claim_id" in matches.columns:
        claim_matches = matches[matches["claim_id"].astype(str) == selected].copy()
        claim_matches = _deduplicate_policy_evidence(claim_matches)

    st.subheader("Claim-level explanation")
    if not final_row.empty:
        risk_level = _safe_first(final_row, ["risk_level", "claim_risk_level"], "—")
        risk_score = _safe_first(final_row, ["risk_score", "claim_risk_score"], None)
        predicted = _safe_first(final_row, ["predicted_denial", "prediction"], "—")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk level", str(risk_level))
        c2.metric("Risk score", _as_percent(risk_score))
        c3.metric("Predicted denial", str(predicted))

        narrative = _safe_first(
            final_row,
            ["final_explanation", "combined_explanation", "narrative", "summary_text"],
            "",
        )
        if narrative:
            with st.expander("Generated narrative", expanded=False):
                st.write(str(narrative))
    else:
        st.caption("Final claim-level summary row was not found for this claim. Showing reason/policy rows only.")

    _display_reasons(claim_matches, final_row)
    _display_policy_evidence(claim_matches)


def _display_reasons(claim_matches: pd.DataFrame, final_row: pd.Series) -> None:
    st.subheader("Reasons")
    reason_rows = _deduplicate_reasons(claim_matches)

    if not reason_rows.empty:
        title_col = _find_column(reason_rows, ["reason_title", "reason", "reason_code"])
        text_col = _find_column(reason_rows, ["reason_text", "business_reason", "reason_description"])
        fix_col = _find_column(reason_rows, ["fix_suggestion", "recommended_fix", "recommended_action"])

        for _, row in reason_rows.head(5).iterrows():
            title = _safe_first(row, [title_col] if title_col else [], "Reason")
            text = _safe_first(row, [text_col] if text_col else [], "")
            fix = _safe_first(row, [fix_col] if fix_col else [], "")
            st.markdown(f"- **{title}**" + (f": {text}" if text else ""))
            if fix:
                st.caption(f"Suggested action: {fix}")
        return

    reasons_text = _safe_first(final_row, ["reason_summary", "reasons", "reason_text"], "")
    if reasons_text:
        st.write(str(reasons_text))
    else:
        st.caption("No reason rows available for this claim.")


def _display_policy_evidence(claim_matches: pd.DataFrame) -> None:
    st.subheader("Policy evidence")
    if claim_matches.empty:
        st.caption("No policy evidence was matched for this claim.")
        return

    source_col = _find_column(claim_matches, ["source_name", "source", "document_id"])
    section_col = _find_column(claim_matches, ["section_title", "section", "heading"])
    score_col = _find_column(claim_matches, ["similarity_score", "score", "retrieval_score", "policy_score"])
    text_col = _find_column(claim_matches, ["policy_text", "chunk_text", "policy_summary", "evidence_text"])
    reason_col = _find_column(claim_matches, ["reason_title", "reason_code", "reason_text"])

    display = claim_matches.copy()
    if text_col:
        display["policy_preview"] = display[text_col].map(_preview_text)
    if score_col:
        display[score_col] = pd.to_numeric(display[score_col], errors="coerce").round(3)

    columns = [col for col in [reason_col, source_col, section_col, score_col, "policy_preview"] if col]
    if columns:
        st.dataframe(display[columns].head(20), width="stretch", hide_index=True)
    else:
        st.dataframe(display.head(20), width="stretch", hide_index=True)

    with st.expander("Detailed policy match rows", expanded=False):
        st.dataframe(display.head(200), width="stretch", hide_index=True)


def _display_data_quality(paths: dict[str, Path], matches: pd.DataFrame) -> None:
    with st.expander("Retrieval quality and diagnostics", expanded=False):
        report = _read_json(paths["report"])
        if report:
            status = report.get("status", "unknown")
            unmatched = report.get("unmatched_reason_count", report.get("unmatched_count", 0))
            st.write(f"Status: `{status}`")
            st.write(f"Unmatched reasons: `{unmatched}`")
        if not matches.empty:
            reason_col = _find_column(matches, ["reason_code", "reason_title"])
            score_col = _find_column(matches, ["similarity_score", "score", "retrieval_score"])
            if reason_col:
                st.write("Policy matches by reason")
                counts = matches[reason_col].value_counts().reset_index()
                counts.columns = ["reason", "match_rows"]
                st.dataframe(counts, width="stretch", hide_index=True)
            if score_col:
                scores = pd.to_numeric(matches[score_col], errors="coerce")
                st.write(
                    {
                        "min_score": round(float(scores.min()), 4) if scores.notna().any() else None,
                        "mean_score": round(float(scores.mean()), 4) if scores.notna().any() else None,
                        "max_score": round(float(scores.max()), 4) if scores.notna().any() else None,
                    }
                )


def render_policy_rag_tab(
    root_dir: Path | str | None = None,
    gold_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    **kwargs: Any,
) -> None:
    """Render the Policy Evidence dashboard tab.

    `models_dir` is accepted for compatibility with `dev_dashboard/app.py`, even
    though this tab currently reads policy artifacts from `data/` and Gold.
    """
    _ = models_dir
    paths = _project_paths(root_dir=root_dir, gold_dir=gold_dir, **kwargs)

    st.header("Policy Evidence — Reason and Policy-Based Explanation")
    st.caption(
        "Connects model-generated risk reasons to policy evidence and suggested actions. "
        "Run the policy ingestion and matching pipeline if any artifact is missing."
    )

    chunks = _read_parquet(paths["chunks"])
    matches = _read_parquet(paths["matches"])
    finals = _read_parquet(paths["finals"])
    matches = _deduplicate_policy_evidence(matches)

    _display_status_cards(paths, chunks, matches, finals)

    with st.expander("Policy chunk table preview", expanded=False):
        if chunks.empty:
            st.info("No policy chunks found. Run `python run_policy_ingest.py`.")
        else:
            st.dataframe(chunks.head(50), width="stretch", hide_index=True)

    st.divider()
    _display_claim_summary(finals, matches)
    st.divider()
    _display_data_quality(paths, matches)


# Backward-compatible aliases for older tab names/imports.
render_policy_evidence_tab = render_policy_rag_tab
render_policy_tab = render_policy_rag_tab


__all__ = ["render_policy_rag_tab", "render_policy_evidence_tab", "render_policy_tab"]
