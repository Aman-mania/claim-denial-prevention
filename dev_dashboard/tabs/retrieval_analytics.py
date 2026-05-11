"""
Retrieval Analytics tab
=======================

Visual analytics for the policy retrieval layer.

This tab reads generated artifacts only:
- data/policies/processed/policy_chunks.parquet
- data/vector_store/policy_vectors.npy
- data/vector_store/policy_metadata.json
- data/gold/gold_claim_explanations.parquet
- data/gold/gold_claim_policy_matches.parquet
- data/gold/gold_claim_final_explanations.parquet

Design principles:
- Always distinguish corpus coverage from retrieval usage.
- Do not hide policy sources that were loaded but not selected as evidence.
- Use deterministic, high-contrast colors across all charts.
- Use unique Streamlit keys and stable renderer signatures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - sklearn is a project dependency; UI still degrades gracefully.
    PCA = None
    TSNE = None


MAX_SCATTER_POINTS = 1500
DEFAULT_TOP_N = 20

# Fixed high-contrast palette. The same source keeps the same color across
# vector, bar, Sankey, and selected-claim charts.
SOURCE_COLOR_SEQUENCE = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#059669",  # green
    "#7C3AED",  # purple
    "#EA580C",  # orange
    "#0891B2",  # cyan
    "#BE123C",  # rose
    "#4F46E5",  # indigo
    "#65A30D",  # lime
    "#9333EA",  # violet
    "#0F766E",  # teal
    "#B45309",  # amber/brown
]
REASON_COLOR = "#64748B"
SELECTED_MATCH_COLOR = "#F59E0B"
MARKER_BORDER_COLOR = "#111827"
ZERO_SOURCE_COLOR = "#CBD5E1"


@dataclass(frozen=True)
class RetrievalAnalyticsPaths:
    root_dir: Path
    gold_dir: Path
    processed_policy_dir: Path
    vector_dir: Path

    @property
    def chunks_path(self) -> Path:
        return self.processed_policy_dir / "policy_chunks.parquet"

    @property
    def vectors_path(self) -> Path:
        return self.vector_dir / "policy_vectors.npy"

    @property
    def vector_metadata_path(self) -> Path:
        return self.vector_dir / "policy_metadata.json"

    @property
    def policy_matches_path(self) -> Path:
        return self.gold_dir / "gold_claim_policy_matches.parquet"

    @property
    def explanations_path(self) -> Path:
        return self.gold_dir / "gold_claim_explanations.parquet"

    @property
    def final_explanations_path(self) -> Path:
        return self.gold_dir / "gold_claim_final_explanations.parquet"

    @property
    def match_report_path(self) -> Path:
        return self.gold_dir / "policy_match_report.json"


def _path(value: str | Path | None, fallback: Path) -> Path:
    return fallback if value is None else Path(value)


def _build_paths(root_dir: str | Path | None = None, gold_dir: str | Path | None = None) -> RetrievalAnalyticsPaths:
    root = _path(root_dir, Path.cwd()).resolve()
    gold = _path(gold_dir, root / "data" / "gold").resolve()
    return RetrievalAnalyticsPaths(
        root_dir=root,
        gold_dir=gold,
        processed_policy_dir=root / "data" / "policies" / "processed",
        vector_dir=root / "data" / "vector_store",
    )


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path_str: str, modified_ns: int | None) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame({"_read_error": [f"{path.name}: {exc}"]})


def _read_parquet(path: Path) -> pd.DataFrame:
    modified_ns = path.stat().st_mtime_ns if path.exists() else None
    return _read_parquet_cached(str(path), modified_ns)


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, modified_ns: int | None) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        return {"_read_error": f"{path.name}: {exc}"}


def _read_json(path: Path) -> dict[str, Any]:
    modified_ns = path.stat().st_mtime_ns if path.exists() else None
    return _read_json_cached(str(path), modified_ns)


@st.cache_data(show_spinner=False)
def _read_vectors_cached(path_str: str, modified_ns: int | None) -> np.ndarray:
    path = Path(path_str)
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    try:
        matrix = np.load(path)
        if matrix.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)
        return matrix.astype(np.float32, copy=False)
    except Exception:
        return np.empty((0, 0), dtype=np.float32)


def _read_vectors(path: Path) -> np.ndarray:
    modified_ns = path.stat().st_mtime_ns if path.exists() else None
    return _read_vectors_cached(str(path), modified_ns)


def _first_present(df: pd.DataFrame, candidates: Iterable[str], default: str | None = None) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _safe_text(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return fallback
    return text


def _shorten(text: Any, max_chars: int = 180) -> str:
    value = _safe_text(text, "")
    return value if len(value) <= max_chars else value[: max_chars - 1].rstrip() + "…"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(100,116,139,{alpha})"
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _source_color_map(values: Iterable[Any]) -> dict[str, str]:
    sources = sorted({_safe_text(v, "Unknown source") for v in values if _safe_text(v, "")})
    return {src: SOURCE_COLOR_SEQUENCE[i % len(SOURCE_COLOR_SEQUENCE)] for i, src in enumerate(sources)}


def _all_source_names(chunks: pd.DataFrame, matches: pd.DataFrame | None = None) -> list[str]:
    values: list[str] = []
    c_source = _source_col(chunks)
    if c_source and not chunks.empty:
        values.extend(chunks[c_source].map(lambda v: _safe_text(v, "Unknown source")).tolist())
    if matches is not None and not matches.empty:
        m_source = _source_col(matches)
        if m_source:
            values.extend(matches[m_source].map(lambda v: _safe_text(v, "Unknown source")).tolist())
    return sorted(set(values))


def _numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_tags(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
    except Exception:
        pass
    return text.replace("[", "").replace("]", "").replace('"', "").replace("'", "")


def _chunk_id_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["policy_chunk_id", "chunk_id", "id", "document_chunk_id"])


def _source_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["source_name", "source", "document_name", "file_name", "document_id"])


def _section_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["section_title", "section", "heading", "title"])


def _score_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["similarity_score", "score", "retrieval_score", "distance"])


def _reason_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["reason_code", "reason_title", "reason", "business_reason"])


def _claim_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["claim_id", "claim"])


def _text_col(df: pd.DataFrame) -> str | None:
    return _first_present(df, ["policy_text", "chunk_text", "text", "policy_summary", "snippet"])


def _dedupe_policy_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    df = matches.copy()
    claim = _claim_col(df)
    reason = _reason_col(df)
    chunk = _chunk_id_col(df)
    source = _source_col(df)
    section = _section_col(df)
    score = _score_col(df)
    subset = [c for c in [claim, reason, chunk, source, section] if c is not None]
    if score:
        df["_score_numeric"] = pd.to_numeric(df[score], errors="coerce").fillna(-1)
        df = df.sort_values("_score_numeric", ascending=False)
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    return df.drop(columns=["_score_numeric"], errors="ignore")


def _prepare_source_utilization(chunks: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Compare available policy-corpus chunks with retrieved evidence by source.

    This function intentionally includes zero-retrieval sources so every loaded
    policy document remains visible in coverage and quality tabs.
    """
    if chunks.empty:
        return pd.DataFrame(columns=["source_name", "policy_chunks", "retrieval_count", "unique_chunks_retrieved", "utilization_rate", "avg_similarity"])

    c_source = _source_col(chunks)
    c_chunk = _chunk_id_col(chunks)
    c = chunks.copy()
    c["source_name"] = c[c_source].map(lambda v: _safe_text(v, "Unknown source")) if c_source else "Unknown source"
    if c_chunk:
        c["policy_chunk_id"] = c[c_chunk].astype(str)
        chunk_counts = c.groupby("source_name", as_index=False)["policy_chunk_id"].nunique().rename(columns={"policy_chunk_id": "policy_chunks"})
    else:
        chunk_counts = c.groupby("source_name", as_index=False).size().rename(columns={"size": "policy_chunks"})

    out = chunk_counts.copy()
    out["retrieval_count"] = 0
    out["unique_chunks_retrieved"] = 0
    out["avg_similarity"] = np.nan

    if not matches.empty:
        m = _dedupe_policy_matches(matches).copy()
        m_source = _source_col(m)
        m_chunk = _chunk_id_col(m)
        m_score = _score_col(m)
        if m_source:
            m["source_name"] = m[m_source].map(lambda v: _safe_text(v, "Unknown source"))
            retrieval_count = m.groupby("source_name", as_index=False).size().rename(columns={"size": "retrieval_count"})
            out = out.drop(columns=["retrieval_count"]).merge(retrieval_count, on="source_name", how="left")
            if m_chunk:
                m["policy_chunk_id"] = m[m_chunk].astype(str)
                unique_retrieved = m.groupby("source_name", as_index=False)["policy_chunk_id"].nunique().rename(columns={"policy_chunk_id": "unique_chunks_retrieved"})
                out = out.drop(columns=["unique_chunks_retrieved"]).merge(unique_retrieved, on="source_name", how="left")
            if m_score:
                avg = m.assign(_score=pd.to_numeric(m[m_score], errors="coerce")).groupby("source_name", as_index=False)["_score"].mean().rename(columns={"_score": "avg_similarity"})
                out = out.drop(columns=["avg_similarity"]).merge(avg, on="source_name", how="left")

    out["retrieval_count"] = out["retrieval_count"].fillna(0).astype(int)
    out["unique_chunks_retrieved"] = out["unique_chunks_retrieved"].fillna(0).astype(int)
    out["avg_similarity"] = pd.to_numeric(out["avg_similarity"], errors="coerce")
    out["utilization_rate"] = np.where(out["policy_chunks"] > 0, out["unique_chunks_retrieved"] / out["policy_chunks"], 0.0)
    return out.sort_values(["retrieval_count", "policy_chunks"], ascending=[False, False]).reset_index(drop=True)


def _prepare_reason_source_flow(
    matches: pd.DataFrame,
    chunks: pd.DataFrame | None = None,
    top_n: int = DEFAULT_TOP_N,
    *,
    include_all_sources: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Aggregate reason -> policy-source counts.

    Backward compatibility note:
    Older dashboard tests and helper callers expect this function to return only
    a DataFrame. The Retrieval Analytics renderer needs the companion corpus
    source list as well, so callers can opt in with ``include_all_sources=True``.
    This keeps the public helper contract stable while supporting the newer
    source-visibility UI.
    """
    all_sources = _all_source_names(chunks if chunks is not None else pd.DataFrame(), matches)
    empty = pd.DataFrame(columns=["reason", "source", "count"])
    if matches.empty:
        return (empty, all_sources) if include_all_sources else empty

    df = _dedupe_policy_matches(matches)
    reason = _reason_col(df)
    source = _source_col(df)
    if not reason or not source:
        return (empty, all_sources) if include_all_sources else empty

    out = (
        df.assign(
            reason=df[reason].map(lambda v: _safe_text(v, "Unknown reason")),
            source=df[source].map(lambda v: _safe_text(v, "Unknown source")),
        )
        .groupby(["reason", "source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return (out, all_sources) if include_all_sources else out


def _prepare_reason_coverage(explanations: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    reason_exp = _reason_col(explanations)
    reason_match = _reason_col(matches)
    claim_exp = _claim_col(explanations)
    claim_match = _claim_col(matches)
    score = _score_col(matches)
    source = _source_col(matches)

    if explanations.empty or not reason_exp:
        return pd.DataFrame(columns=["reason_code", "reason_rows", "matched_rows", "match_rate", "avg_similarity", "top_policy_source"])

    exp = explanations.copy()
    exp["reason_code"] = exp[reason_exp].map(lambda v: _safe_text(v, "Unknown reason"))
    exp["_reason_key"] = exp[claim_exp].astype(str) + "||" + exp["reason_code"] if claim_exp else exp["reason_code"]
    reason_rows = exp.groupby("reason_code", as_index=False)["_reason_key"].nunique().rename(columns={"_reason_key": "reason_rows"})

    if matches.empty or not reason_match:
        reason_rows["matched_rows"] = 0
        reason_rows["match_rate"] = 0.0
        reason_rows["avg_similarity"] = np.nan
        reason_rows["top_policy_source"] = "—"
        return reason_rows.sort_values("reason_rows", ascending=False)

    m = _dedupe_policy_matches(matches)
    m["reason_code"] = m[reason_match].map(lambda v: _safe_text(v, "Unknown reason"))
    m["_reason_key"] = m[claim_match].astype(str) + "||" + m["reason_code"] if claim_match else m["reason_code"]
    matched_rows = m.groupby("reason_code", as_index=False)["_reason_key"].nunique().rename(columns={"_reason_key": "matched_rows"})

    avg = pd.DataFrame({"reason_code": matched_rows["reason_code"], "avg_similarity": np.nan})
    if score:
        avg = m.assign(_score=pd.to_numeric(m[score], errors="coerce")).groupby("reason_code", as_index=False)["_score"].mean().rename(columns={"_score": "avg_similarity"})

    if source:
        src = (
            m.assign(_source=m[source].map(lambda v: _safe_text(v, "Unknown source")))
            .groupby(["reason_code", "_source"], as_index=False)
            .size()
            .sort_values(["reason_code", "size"], ascending=[True, False])
            .drop_duplicates("reason_code")
            .rename(columns={"_source": "top_policy_source"})[["reason_code", "top_policy_source"]]
        )
    else:
        src = pd.DataFrame({"reason_code": matched_rows["reason_code"], "top_policy_source": "—"})

    out = reason_rows.merge(matched_rows, on="reason_code", how="left").merge(avg, on="reason_code", how="left").merge(src, on="reason_code", how="left")
    out["matched_rows"] = out["matched_rows"].fillna(0).astype(int)
    out["match_rate"] = np.where(out["reason_rows"] > 0, out["matched_rows"] / out["reason_rows"], 0.0)
    out["top_policy_source"] = out["top_policy_source"].fillna("—")
    return out.sort_values(["match_rate", "reason_rows"], ascending=[True, False])


def _source_summary_chart(util: pd.DataFrame, metric: str, title: str, key: str, color_map: dict[str, str], log_x: bool = False) -> None:
    if util.empty or metric not in util.columns:
        st.info(f"{title} cannot be shown yet.")
        return
    chart_df = util.sort_values(metric, ascending=True).copy()
    chart_df["bar_color"] = chart_df["source_name"].map(lambda s: color_map.get(str(s), ZERO_SOURCE_COLOR))
    fig = go.Figure(
        go.Bar(
            y=chart_df["source_name"],
            x=chart_df[metric],
            orientation="h",
            marker=dict(color=chart_df["bar_color"], line=dict(color="#111827", width=0.3)),
            text=chart_df[metric].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—"),
            textposition="outside",
            hovertemplate="%{y}<br>" + title + ": %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(330, 58 * len(chart_df)),
        margin=dict(l=10, r=70, t=50, b=20),
        xaxis_title="Count",
        yaxis_title="Policy source",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    if log_x and chart_df[metric].max() > 0:
        fig.update_xaxes(type="log")
    st.plotly_chart(fig, width="stretch", key=key)


def _render_source_presence(util: pd.DataFrame, key_prefix: str, compact: bool = False) -> None:
    """Reusable source visibility block for tabs where retrieval-only charts can hide sources."""
    if util.empty:
        return
    color_map = _source_color_map(util["source_name"].tolist())
    st.caption("Corpus coverage is shown separately from selected evidence so underused sources remain visible.")
    if compact:
        c1, c2 = st.columns(2)
        with c1:
            _source_summary_chart(util, "policy_chunks", "Policy chunks in corpus", f"{key_prefix}_corpus_chunks", color_map)
        with c2:
            _source_summary_chart(util, "retrieval_count", "Retrieved evidence rows", f"{key_prefix}_retrieved_rows", color_map, log_x=bool(util["retrieval_count"].max() > 5000))
    else:
        _source_summary_chart(util, "policy_chunks", "Policy chunks in corpus", f"{key_prefix}_corpus_chunks", color_map)
        _source_summary_chart(util, "retrieval_count", "Retrieved evidence rows", f"{key_prefix}_retrieved_rows", color_map, log_x=bool(util["retrieval_count"].max() > 5000))


def _render_source_utilization(chunks: pd.DataFrame, matches: pd.DataFrame) -> None:
    st.subheader("Policy source coverage")
    util = _prepare_source_utilization(chunks, matches)
    if util.empty:
        st.info("No policy source coverage can be shown yet.")
        return

    source_count = len(util)
    zero_count = int((util["retrieval_count"] == 0).sum())
    avg_util = float(util["utilization_rate"].mean()) if source_count else 0.0
    cols = st.columns(4)
    cols[0].metric("Policy sources", f"{source_count:,}")
    cols[1].metric("Sources retrieved", f"{source_count - zero_count:,}")
    cols[2].metric("Sources not selected", f"{zero_count:,}")
    cols[3].metric("Avg chunk utilization", f"{avg_util * 100:.1f}%")

    c1, c2 = st.columns(2)
    color_map = _source_color_map(util["source_name"].tolist())
    with c1:
        _source_summary_chart(util, "policy_chunks", "Policy chunks in corpus", "retrieval_source_chunks_bar", color_map)
    with c2:
        _source_summary_chart(util, "retrieval_count", "Retrieved evidence rows", "retrieval_source_evidence_bar", color_map, log_x=bool(util["retrieval_count"].max() > 5000))

    util_chart = util.sort_values("utilization_rate", ascending=True).copy()
    util_chart["util_pct"] = util_chart["utilization_rate"] * 100
    fig = go.Figure(
        go.Bar(
            y=util_chart["source_name"],
            x=util_chart["util_pct"],
            orientation="h",
            marker=dict(color=util_chart["source_name"].map(lambda s: color_map.get(str(s), ZERO_SOURCE_COLOR)), line=dict(color="#111827", width=0.3)),
            text=util_chart["util_pct"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="%{y}<br>Unique retrieved chunks / total chunks: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Unique chunk utilization by source",
        height=max(330, 58 * len(util_chart)),
        margin=dict(l=10, r=70, t=50, b=20),
        xaxis_title="Utilization (%)",
        yaxis_title="Policy source",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", key="retrieval_source_utilization_pct")

    if zero_count:
        st.caption(
            "Some policy sources are present in the vector index but were not selected as top evidence for the current reason distribution. "
            "This is a retrieval-quality signal, not a missing-document problem."
        )
    with st.expander("Policy source coverage table", expanded=False):
        display = util.copy()
        display["utilization_rate"] = (display["utilization_rate"] * 100).round(2).astype(str) + "%"
        display["avg_similarity"] = pd.to_numeric(display["avg_similarity"], errors="coerce").round(4)
        st.dataframe(display, width="stretch", hide_index=True)


def _prepare_top_chunks(matches: pd.DataFrame, chunks: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["policy_chunk_id", "retrieval_count", "avg_similarity", "source_name", "section_title", "snippet"])
    df = _dedupe_policy_matches(matches)
    chunk = _chunk_id_col(df)
    score = _score_col(df)
    if not chunk:
        return pd.DataFrame(columns=["policy_chunk_id", "retrieval_count", "avg_similarity", "source_name", "section_title", "snippet"])
    agg = df.assign(policy_chunk_id=df[chunk].astype(str)).groupby("policy_chunk_id", as_index=False).size().rename(columns={"size": "retrieval_count"})
    if score:
        avg = df.assign(policy_chunk_id=df[chunk].astype(str), _score=pd.to_numeric(df[score], errors="coerce")).groupby("policy_chunk_id", as_index=False)["_score"].mean().rename(columns={"_score": "avg_similarity"})
        agg = agg.merge(avg, on="policy_chunk_id", how="left")
    else:
        agg["avg_similarity"] = np.nan
    if not chunks.empty:
        c_chunk = _chunk_id_col(chunks)
        if c_chunk:
            meta = chunks.copy()
            meta["policy_chunk_id"] = meta[c_chunk].astype(str)
            s_col = _source_col(meta)
            sec_col = _section_col(meta)
            txt_col = _text_col(meta)
            keep = ["policy_chunk_id"]
            if s_col:
                meta["source_name"] = meta[s_col].map(_safe_text)
                keep.append("source_name")
            if sec_col:
                meta["section_title"] = meta[sec_col].map(_safe_text)
                keep.append("section_title")
            if txt_col:
                meta["snippet"] = meta[txt_col].map(lambda v: _shorten(v, 220))
                keep.append("snippet")
            agg = agg.merge(meta[keep].drop_duplicates("policy_chunk_id"), on="policy_chunk_id", how="left")
    for col in ["source_name", "section_title", "snippet"]:
        if col not in agg.columns:
            agg[col] = "—"
    return agg.sort_values("retrieval_count", ascending=False).head(top_n)


def _sample_vectors(vectors: np.ndarray, chunks: pd.DataFrame, max_points: int = MAX_SCATTER_POINTS) -> tuple[np.ndarray, pd.DataFrame]:
    if vectors.size == 0 or len(chunks) == 0:
        return np.empty((0, 0), dtype=np.float32), chunks.iloc[0:0].copy()
    n = min(vectors.shape[0], len(chunks))
    vec = vectors[:n]
    meta = chunks.iloc[:n].copy().reset_index(drop=True)
    if n <= max_points:
        return vec, meta
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return vec[idx], meta.iloc[idx].reset_index(drop=True)


def _project_vectors(vectors: np.ndarray, method: str = "PCA") -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    method = (method or "PCA").upper()
    safe_vectors = np.nan_to_num(vectors.astype(np.float32, copy=False))
    if method == "T-SNE" and TSNE is not None and safe_vectors.shape[0] >= 4:
        perplexity = min(30, max(2, (safe_vectors.shape[0] - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(safe_vectors)
    if PCA is None:
        return safe_vectors[:, :2]
    return PCA(n_components=2, random_state=42).fit_transform(safe_vectors)


def _prepare_vector_projection(vectors: np.ndarray, chunks: pd.DataFrame, matches: pd.DataFrame, selected_claim: str | None = None, method: str = "PCA") -> pd.DataFrame:
    vec, meta = _sample_vectors(vectors, chunks)
    if vec.size == 0 or meta.empty:
        return pd.DataFrame(columns=["x", "y", "policy_chunk_id", "source_name", "section_title", "retrieval_count", "selected_claim_match", "hover"])
    projection = _project_vectors(vec, method=method)
    if projection.size == 0:
        return pd.DataFrame(columns=["x", "y", "policy_chunk_id", "source_name", "section_title", "retrieval_count", "selected_claim_match", "hover"])
    out = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1]})
    c_chunk = _chunk_id_col(meta)
    src = _source_col(meta)
    sec = _section_col(meta)
    txt = _text_col(meta)
    tag_col = _first_present(meta, ["policy_tags", "tags", "tag"])
    out["policy_chunk_id"] = meta[c_chunk].astype(str).values if c_chunk else [f"chunk_{i}" for i in range(len(out))]
    out["source_name"] = meta[src].map(_safe_text).values if src else "Unknown source"
    out["section_title"] = meta[sec].map(_safe_text).values if sec else "Unknown section"
    out["snippet"] = meta[txt].map(lambda v: _shorten(v, 180)).values if txt else ""
    out["tags"] = meta[tag_col].map(_normalize_tags).values if tag_col else ""

    if not matches.empty:
        m = _dedupe_policy_matches(matches)
        m_chunk = _chunk_id_col(m)
        m_claim = _claim_col(m)
        m_reason = _reason_col(m)
        m_score = _score_col(m)
        if m_chunk:
            count = m.assign(policy_chunk_id=m[m_chunk].astype(str)).groupby("policy_chunk_id").size().rename("retrieval_count")
            out = out.merge(count, on="policy_chunk_id", how="left")
            if selected_claim and m_claim:
                selected = m[m[m_claim].astype(str) == str(selected_claim)].copy()
                selected_ids = set(selected[m_chunk].astype(str).tolist())
                out["selected_claim_match"] = out["policy_chunk_id"].isin(selected_ids)
                if m_reason and m_score and not selected.empty:
                    selected_detail = selected.assign(policy_chunk_id=selected[m_chunk].astype(str), _reason=selected[m_reason].map(_safe_text), _score=pd.to_numeric(selected[m_score], errors="coerce"))
                    summary_rows = []
                    for chunk_id, group in selected_detail.groupby("policy_chunk_id", sort=False):
                        parts = []
                        for r, score_value in zip(group["_reason"].head(3), group["_score"].head(3)):
                            parts.append(f"{r} ({float(score_value):.2f})" if pd.notna(score_value) else str(r))
                        summary_rows.append({"policy_chunk_id": chunk_id, "selected_claim_reason": "; ".join(parts)})
                    reason_summary = pd.DataFrame(summary_rows).set_index("policy_chunk_id")["selected_claim_reason"] if summary_rows else pd.Series(dtype="object", name="selected_claim_reason")
                    out = out.merge(reason_summary, on="policy_chunk_id", how="left")
            else:
                out["selected_claim_match"] = False
    if "retrieval_count" not in out.columns:
        out["retrieval_count"] = 0
    if "selected_claim_match" not in out.columns:
        out["selected_claim_match"] = False
    if "selected_claim_reason" not in out.columns:
        out["selected_claim_reason"] = ""
    out["retrieval_count"] = out["retrieval_count"].fillna(0).astype(int)
    out["selected_claim_reason"] = out["selected_claim_reason"].fillna("")
    out["hover"] = (
        "Chunk: " + out["policy_chunk_id"].astype(str)
        + "<br>Source: " + out["source_name"].astype(str)
        + "<br>Section: " + out["section_title"].astype(str)
        + "<br>Retrieved: " + out["retrieval_count"].astype(str)
        + np.where(out["selected_claim_reason"].astype(str).str.len() > 0, "<br>Selected claim: " + out["selected_claim_reason"].astype(str), "")
        + "<br>" + out["snippet"].astype(str)
    )
    return out


def _prepare_shap_similarity(explanations: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if explanations.empty or matches.empty:
        return pd.DataFrame(columns=["claim_id", "reason_code", "shap_value", "similarity_score", "risk_level", "source_name"])
    claim_e = _claim_col(explanations)
    reason_e = _reason_col(explanations)
    claim_m = _claim_col(matches)
    reason_m = _reason_col(matches)
    score = _score_col(matches)
    source = _source_col(matches)
    shap_col = _first_present(explanations, ["shap_value", "shap_contribution", "raw_log_odds_contribution"])
    if not all([claim_e, reason_e, claim_m, reason_m, score, shap_col]):
        return pd.DataFrame(columns=["claim_id", "reason_code", "shap_value", "similarity_score", "risk_level", "source_name"])
    e = explanations.copy()
    e["claim_id"] = e[claim_e].astype(str)
    e["reason_code"] = e[reason_e].map(_safe_text)
    e["shap_value"] = pd.to_numeric(e[shap_col], errors="coerce")
    risk_col = _first_present(e, ["risk_level", "risk_band"])
    e["risk_level"] = e[risk_col].map(_safe_text) if risk_col else "Unknown"
    m = _dedupe_policy_matches(matches).copy()
    m["claim_id"] = m[claim_m].astype(str)
    m["reason_code"] = m[reason_m].map(_safe_text)
    m["similarity_score"] = pd.to_numeric(m[score], errors="coerce")
    if source:
        m["source_name"] = m[source].map(_safe_text)
    else:
        m["source_name"] = "Unknown source"
    m = m.sort_values("similarity_score", ascending=False).drop_duplicates(["claim_id", "reason_code"], keep="first")
    out = e[["claim_id", "reason_code", "shap_value", "risk_level"]].merge(m[["claim_id", "reason_code", "similarity_score", "source_name"]], on=["claim_id", "reason_code"], how="inner")
    return out.dropna(subset=["shap_value", "similarity_score"])


def _render_missing_artifacts(paths: RetrievalAnalyticsPaths) -> bool:
    required = {"Policy chunks": paths.chunks_path, "Policy matches": paths.policy_matches_path, "Claim explanations": paths.explanations_path}
    missing = {name: path for name, path in required.items() if not path.exists()}
    if not missing:
        return False
    st.warning("Retrieval analytics artifacts are not complete yet.")
    rows = [{"artifact": name, "expected_path": str(path), "status": "missing"} for name, path in missing.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(
        """
Run the policy explanation flow first:

```bash
python run_explain.py
python run_policy_ingest.py
python run_policy_match.py
```

or:

```bash
python run_week6.py --mode preferred
```
        """
    )
    return True


def _render_overview(chunks: pd.DataFrame, vectors: np.ndarray, matches: pd.DataFrame, explanations: pd.DataFrame, final_outputs: pd.DataFrame, metadata: dict[str, Any]) -> None:
    score = _score_col(matches)
    avg_score = float(pd.to_numeric(matches[score], errors="coerce").mean()) if score and not matches.empty else np.nan
    claim = _claim_col(final_outputs)
    final_claims = final_outputs[claim].nunique() if claim and not final_outputs.empty else 0
    reason = _reason_col(explanations)
    reason_rows = len(explanations)
    reason_types = explanations[reason].nunique() if reason and not explanations.empty else 0
    cols = st.columns(6)
    cols[0].metric("Policy chunks", f"{len(chunks):,}")
    cols[1].metric("Policy sources", f"{len(_all_source_names(chunks, matches)):,}")
    cols[2].metric("Embedding dim", f"{vectors.shape[1] if vectors.ndim == 2 and vectors.size else metadata.get('embedding_dim', '—')}")
    cols[3].metric("Reason types", f"{reason_types:,}")
    cols[4].metric("Policy matches", f"{len(matches):,}")
    cols[5].metric("Claims explained", f"{final_claims:,}")
    meta_cols = st.columns(3)
    embedding_backend = metadata.get("embedding_backend") or metadata.get("backend") or metadata.get("embedding", {}).get("backend") or "Unknown"
    vector_backend = metadata.get("vector_backend") or metadata.get("index_backend") or metadata.get("backend_type") or "Unknown"
    embedding_model = metadata.get("embedding_model") or metadata.get("model_name") or metadata.get("embedding", {}).get("model_name") or "Unknown"
    meta_cols[0].caption(f"Embedding backend: `{embedding_backend}`")
    meta_cols[1].caption(f"Vector backend: `{vector_backend}`")
    meta_cols[2].caption(f"Embedding model: `{embedding_model}`")
    if not np.isnan(avg_score):
        st.caption(f"Average similarity score across policy matches: `{avg_score:.4f}` · Reason rows: `{reason_rows:,}`")


def _render_vector_space(chunks: pd.DataFrame, vectors: np.ndarray, matches: pd.DataFrame) -> None:
    st.subheader("Vector space explorer")
    if vectors.size == 0:
        st.info("No vector matrix found. Run policy ingestion first to create `data/vector_store/policy_vectors.npy`.")
        return
    if chunks.empty:
        st.info("No policy chunk metadata found. Run policy ingestion first.")
        return
    claim = _claim_col(matches)
    claim_ids = sorted(matches[claim].dropna().astype(str).unique().tolist()) if claim and not matches.empty else []
    controls = st.columns([1, 1, 2])
    method = controls[0].selectbox("Projection", ["PCA", "t-SNE"], index=0, key="retrieval_projection_method")
    selected_claim = None
    if claim_ids:
        selected_claim = controls[1].selectbox("Highlight claim", ["None"] + claim_ids, index=0, key="retrieval_vector_claim_select")
        selected_claim = None if selected_claim == "None" else selected_claim
    else:
        controls[1].caption("No claim matches available for highlighting.")
    controls[2].caption("Each point is a policy chunk. Larger points are reused more often; highlighted diamonds were retrieved for the selected claim.")
    scatter = _prepare_vector_projection(vectors, chunks, matches, selected_claim=selected_claim, method=method)
    if scatter.empty:
        st.info("Vector projection could not be created. Check that vectors and policy chunks have at least two rows.")
        return
    color_map = _source_color_map(scatter["source_name"].tolist())
    fig = go.Figure()
    for source_name, group in scatter[~scatter["selected_claim_match"]].groupby("source_name", sort=True):
        size = np.clip(13 + np.log1p(group["retrieval_count"].astype(float)) * 3.5, 13, 28)
        fig.add_trace(
            go.Scatter(
                x=group["x"], y=group["y"], mode="markers", name=str(source_name), customdata=group["hover"], hovertemplate="%{customdata}<extra></extra>",
                marker=dict(size=size, color=color_map.get(str(source_name), "#2563EB"), opacity=0.96, line=dict(color=MARKER_BORDER_COLOR, width=1.2), symbol="circle"),
            )
        )
    selected_group = scatter[scatter["selected_claim_match"]]
    if not selected_group.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_group["x"], y=selected_group["y"], mode="markers", name="Selected claim evidence", customdata=selected_group["hover"], hovertemplate="%{customdata}<extra></extra>",
                marker=dict(size=28, color=SELECTED_MATCH_COLOR, opacity=1.0, line=dict(color="#000000", width=2.5), symbol="diamond"),
            )
        )
    x_min, x_max = scatter["x"].min(), scatter["x"].max()
    y_min, y_max = scatter["y"].min(), scatter["y"].max()
    x_pad = max((x_max - x_min) * 0.18, 0.03)
    y_pad = max((y_max - y_min) * 0.18, 0.03)
    fig.update_layout(
        title=f"Policy chunk vector projection ({method})",
        height=650,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Policy source",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Projection axis 1", range=[x_min - x_pad, x_max + x_pad], showgrid=True, gridcolor="#E5E7EB", zeroline=True, zerolinecolor="#CBD5E1"),
        yaxis=dict(title="Projection axis 2", range=[y_min - y_pad, y_max + y_pad], showgrid=True, gridcolor="#E5E7EB", zeroline=True, zerolinecolor="#CBD5E1"),
    )
    st.plotly_chart(fig, width="stretch", key="retrieval_vector_space_scatter")
    util = _prepare_source_utilization(chunks, matches)
    with st.expander("Source visibility check", expanded=False):
        _render_source_presence(util, key_prefix="retrieval_vector_source_presence", compact=True)
    if selected_claim:
        selected_rows = scatter[scatter["selected_claim_match"]]
        st.caption(f"Highlighted chunks for claim `{selected_claim}`: {len(selected_rows)}")
        if not selected_rows.empty:
            st.dataframe(selected_rows[["policy_chunk_id", "source_name", "section_title", "retrieval_count", "selected_claim_reason", "snippet"]].head(20), width="stretch", hide_index=True)


def _render_similarity_analytics(matches: pd.DataFrame, explanations: pd.DataFrame, chunks: pd.DataFrame) -> None:
    st.subheader("Similarity score quality")
    score = _score_col(matches)
    reason = _reason_col(matches)
    source = _source_col(matches)
    util = _prepare_source_utilization(chunks, matches)
    if matches.empty or not score:
        st.info("No similarity scores are available in policy matches.")
        _render_source_presence(util, key_prefix="retrieval_score_quality_sources", compact=True)
        return
    df = _dedupe_policy_matches(matches).copy()
    df["_score"] = pd.to_numeric(df[score], errors="coerce")
    df = df.dropna(subset=["_score"])
    if df.empty:
        st.info("Policy match rows exist, but similarity score values are not numeric.")
        _render_source_presence(util, key_prefix="retrieval_score_quality_sources_non_numeric", compact=True)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min score", f"{float(df['_score'].min()):.3f}")
    c2.metric("Average score", f"{float(df['_score'].mean()):.3f}")
    c3.metric("Max score", f"{float(df['_score'].max()):.3f}")
    c4.metric("Sources retrieved", f"{int((util['retrieval_count'] > 0).sum()) if not util.empty else 0:,}")
    chart_cols = st.columns([1.1, 1])
    with chart_cols[0]:
        fig = px.histogram(df, x="_score", nbins=30, labels={"_score": "Similarity score"}, title="Similarity score distribution")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=390, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width="stretch", key="retrieval_similarity_histogram")
    with chart_cols[1]:
        if reason:
            top_reasons = df[reason].astype(str).value_counts().head(12).index
            box_df = df[df[reason].astype(str).isin(top_reasons)].copy()
            box_df["reason"] = box_df[reason].astype(str)
            fig = px.box(box_df, x="reason", y="_score", labels={"reason": "Reason", "_score": "Similarity"}, title="Similarity by reason")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=390, xaxis_tickangle=-35, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width="stretch", key="retrieval_similarity_by_reason")
        elif source:
            top_sources = df[source].astype(str).value_counts().head(12).index
            box_df = df[df[source].astype(str).isin(top_sources)].copy()
            box_df["source"] = box_df[source].astype(str)
            fig = px.box(box_df, x="source", y="_score", labels={"source": "Source", "_score": "Similarity"}, title="Similarity by source")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=390, xaxis_tickangle=-35, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width="stretch", key="retrieval_similarity_by_source")
    st.markdown("#### Source-level retrieval counts")
    _render_source_presence(util, key_prefix="retrieval_similarity_sources", compact=True)
    coverage = _prepare_reason_coverage(explanations, matches)
    if not coverage.empty:
        with st.expander("Reason coverage table", expanded=False):
            display = coverage.copy()
            display["match_rate"] = (display["match_rate"] * 100).round(2).astype(str) + "%"
            display["avg_similarity"] = pd.to_numeric(display["avg_similarity"], errors="coerce").round(4)
            st.dataframe(display, width="stretch", hide_index=True)


def _render_reason_policy_flow(matches: pd.DataFrame, chunks: pd.DataFrame) -> None:
    st.subheader("Reason-to-policy flow")
    flow, all_sources = _prepare_reason_source_flow(matches, chunks, top_n=40, include_all_sources=True)
    util = _prepare_source_utilization(chunks, matches)
    if flow.empty:
        st.info("Reason-to-policy flow cannot be built because reason/source columns are missing or no matches were generated.")
        _render_source_presence(util, key_prefix="retrieval_reason_flow_sources_empty", compact=True)
        return
    reasons = flow["reason"].unique().tolist()
    matched_sources = flow["source"].unique().tolist()
    # Include all corpus sources as right-side nodes so underused policy docs remain visible.
    sources = sorted(set(all_sources) | set(matched_sources))
    labels = reasons + sources
    label_index = {label: i for i, label in enumerate(labels)}
    color_map = _source_color_map(sources)
    node_colors = [REASON_COLOR] * len(reasons) + [color_map.get(s, ZERO_SOURCE_COLOR) for s in sources]
    source_idx = [label_index[r] for r in flow["reason"]]
    target_idx = [label_index[s] for s in flow["source"]]
    values = flow["count"].astype(int).tolist()
    link_colors = [_hex_to_rgba(color_map.get(s, "#64748B"), 0.35) for s in flow["source"]]
    fig = go.Figure(
        data=[go.Sankey(
            node=dict(label=labels, pad=20, thickness=17, line=dict(color="#111827", width=0.4), color=node_colors),
            link=dict(source=source_idx, target=target_idx, value=values, color=link_colors),
        )]
    )
    fig.update_layout(title="Business reasons mapped to policy sources", height=560, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=11))
    st.plotly_chart(fig, width="stretch", key="retrieval_reason_policy_sankey")
    zero_sources = util[util["retrieval_count"] == 0]["source_name"].tolist() if not util.empty else []
    if zero_sources:
        st.caption("Sources with no selected evidence in this run: " + ", ".join(zero_sources))
    with st.expander("Source coverage beside flow", expanded=False):
        _render_source_presence(util, key_prefix="retrieval_reason_flow_sources", compact=True)


def _render_top_chunks(matches: pd.DataFrame, chunks: pd.DataFrame) -> None:
    st.subheader("Most reused policy chunks")
    top_chunks = _prepare_top_chunks(matches, chunks, top_n=20)
    util = _prepare_source_utilization(chunks, matches)
    if top_chunks.empty:
        st.info("Policy chunk frequency cannot be shown yet.")
        _render_source_presence(util, key_prefix="retrieval_top_chunks_sources_empty", compact=True)
        return
    color_map = _source_color_map(_all_source_names(chunks, matches))
    label = top_chunks["section_title"].where(top_chunks["section_title"].astype(str).ne("—"), top_chunks["policy_chunk_id"])
    chart_df = top_chunks.assign(label=label.map(lambda v: _shorten(v, 70))).sort_values("retrieval_count", ascending=True)
    chart_df["bar_color"] = chart_df["source_name"].map(lambda s: color_map.get(str(s), ZERO_SOURCE_COLOR))
    fig = go.Figure(go.Bar(
        x=chart_df["retrieval_count"],
        y=chart_df["label"],
        orientation="h",
        marker=dict(color=chart_df["bar_color"], line=dict(color="#111827", width=0.3)),
        customdata=np.stack([chart_df["policy_chunk_id"], chart_df["source_name"], chart_df["avg_similarity"].fillna(np.nan), chart_df["snippet"]], axis=-1),
        hovertemplate="Chunk: %{customdata[0]}<br>Source: %{customdata[1]}<br>Retrievals: %{x}<br>Avg similarity: %{customdata[2]:.3f}<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Retrieval count", yaxis_title="Policy chunk", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch", key="retrieval_top_chunks_bar")
    with st.expander("Source coverage for top-chunk context", expanded=False):
        _render_source_presence(util, key_prefix="retrieval_top_chunks_sources", compact=True)
    with st.expander("Top retrieved chunk details", expanded=False):
        display = top_chunks.copy()
        display["avg_similarity"] = pd.to_numeric(display["avg_similarity"], errors="coerce").round(4)
        st.dataframe(display, width="stretch", hide_index=True)


def _render_claim_retrieval_graph(matches: pd.DataFrame, chunks: pd.DataFrame) -> None:
    st.subheader("Claim-level retrieval graph")
    claim = _claim_col(matches)
    reason = _reason_col(matches)
    source = _source_col(matches)
    score = _score_col(matches)
    util_all = _prepare_source_utilization(chunks, matches)
    if matches.empty or not all([claim, reason, source]):
        st.info("Claim-level graph requires claim, reason, and source columns in policy matches.")
        _render_source_presence(util_all, key_prefix="retrieval_claim_graph_sources_empty", compact=True)
        return
    claim_ids = sorted(matches[claim].dropna().astype(str).unique().tolist())
    if not claim_ids:
        st.info("No claim IDs available for retrieval graph.")
        return
    selected_claim = st.selectbox("Select claim for retrieval graph", claim_ids, index=0, key="retrieval_graph_claim_select")
    selected_matches = matches[matches[claim].astype(str) == str(selected_claim)].copy()
    m = _dedupe_policy_matches(selected_matches)
    if m.empty:
        st.info("No policy evidence found for the selected claim.")
        return
    flow = pd.DataFrame({"reason": m[reason].map(lambda v: _safe_text(v, "Unknown reason")), "source": m[source].map(lambda v: _safe_text(v, "Unknown source"))})
    flow = flow.groupby(["reason", "source"], as_index=False).size().rename(columns={"size": "count"})
    reasons = flow["reason"].unique().tolist()
    all_sources = _all_source_names(chunks, matches)
    sources = sorted(set(all_sources) | set(flow["source"].unique().tolist()))
    labels = [f"Reason: {r}" for r in reasons] + [f"Policy: {s}" for s in sources]
    reason_idx = {r: i for i, r in enumerate(reasons)}
    source_idx = {s: len(reasons) + i for i, s in enumerate(sources)}
    color_map = _source_color_map(sources)
    node_colors = [REASON_COLOR] * len(reasons) + [color_map.get(s, ZERO_SOURCE_COLOR) for s in sources]
    link_colors = [_hex_to_rgba(color_map.get(s, "#64748B"), 0.38) for s in flow["source"]]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=18, thickness=16, line=dict(color="#111827", width=0.4), color=node_colors),
        link=dict(source=[reason_idx[r] for r in flow["reason"]], target=[source_idx[s] for s in flow["source"]], value=flow["count"].astype(int).tolist(), color=link_colors),
    )])
    fig.update_layout(title=f"Claim {selected_claim}: reason-to-policy retrieval path", height=450, margin=dict(l=10, r=10, t=50, b=10), font=dict(size=11))
    st.plotly_chart(fig, width="stretch", key="retrieval_claim_graph_sankey")
    # Show selected-claim source coverage separately because a single claim will
    # naturally use only a subset of the corpus. This prevents non-selected docs
    # from seeming missing.
    selected_util = _prepare_source_utilization(chunks, selected_matches)
    with st.expander("Selected-claim source coverage", expanded=True):
        _render_source_presence(selected_util, key_prefix="retrieval_claim_selected_sources", compact=True)


def _render_shap_similarity(explanations: pd.DataFrame, matches: pd.DataFrame, chunks: pd.DataFrame) -> None:
    st.subheader("Model reason strength vs policy similarity")
    df = _prepare_shap_similarity(explanations, matches)
    util = _prepare_source_utilization(chunks, matches)
    if df.empty:
        st.info("This chart requires SHAP values in explanations and similarity scores in policy matches.")
        _render_source_presence(util, key_prefix="retrieval_shap_sources_empty", compact=True)
        return
    top_reasons = df["reason_code"].value_counts().head(12).index
    view = df[df["reason_code"].isin(top_reasons)].copy()
    color_map = _source_color_map(_all_source_names(chunks, matches))
    fig = px.scatter(
        view,
        x="shap_value",
        y="similarity_score",
        color="source_name",
        color_discrete_map=color_map,
        symbol="reason_code",
        hover_data=["claim_id", "risk_level", "reason_code"],
        labels={"shap_value": "SHAP contribution", "similarity_score": "Best policy similarity", "source_name": "Policy source", "reason_code": "Reason"},
        title="Higher-right points indicate strong model reasons with strong policy evidence",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.82, line=dict(color="#111827", width=0.4)))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch", key="retrieval_shap_similarity_scatter")
    with st.expander("Policy source coverage for SHAP chart", expanded=False):
        _render_source_presence(util, key_prefix="retrieval_shap_sources", compact=True)


def render_retrieval_analytics_tab(root_dir: str | Path | None = None, gold_dir: str | Path | None = None, models_dir: str | Path | None = None, **kwargs: Any) -> None:
    """Render retrieval analytics dashboard tab.

    Signature accepts root_dir/gold_dir/models_dir/**kwargs so app.py can call
    all dashboard tabs with a common renderer contract.
    """
    paths = _build_paths(root_dir=root_dir, gold_dir=gold_dir)
    st.header("Policy Retrieval Analytics")
    st.caption("Visual diagnostics for policy evidence retrieval, vector search, and reason-to-policy matching.")
    if _render_missing_artifacts(paths):
        return
    chunks = _read_parquet(paths.chunks_path)
    matches = _read_parquet(paths.policy_matches_path)
    explanations = _read_parquet(paths.explanations_path)
    finals = _read_parquet(paths.final_explanations_path)
    metadata = _read_json(paths.vector_metadata_path)
    vectors = _read_vectors(paths.vectors_path)
    read_errors = []
    for name, df in [("policy chunks", chunks), ("policy matches", matches), ("claim explanations", explanations), ("final outputs", finals)]:
        if "_read_error" in df.columns:
            read_errors.append(df["_read_error"].iloc[0])
    if metadata.get("_read_error"):
        read_errors.append(metadata["_read_error"])
    if read_errors:
        st.error("Some retrieval artifacts could not be read.")
        st.write(read_errors)
        return
    _render_overview(chunks, vectors, matches, explanations, finals, metadata)
    tabs = st.tabs(["Vector space", "Source coverage", "Score quality", "Reason-policy flow", "Claim graph", "SHAP vs policy", "Tables"])
    with tabs[0]:
        _render_vector_space(chunks, vectors, matches)
    with tabs[1]:
        _render_source_utilization(chunks, matches)
    with tabs[2]:
        _render_similarity_analytics(matches, explanations, chunks)
    with tabs[3]:
        _render_reason_policy_flow(matches, chunks)
        st.divider()
        _render_top_chunks(matches, chunks)
    with tabs[4]:
        _render_claim_retrieval_graph(matches, chunks)
    with tabs[5]:
        _render_shap_similarity(explanations, matches, chunks)
    with tabs[6]:
        st.subheader("Generated retrieval tables")
        table_choice = st.selectbox("Select table", ["Policy chunks", "Policy matches", "Claim explanations", "Final explanations"], key="retrieval_analytics_table_select")
        table_map = {"Policy chunks": chunks, "Policy matches": _dedupe_policy_matches(matches), "Claim explanations": explanations, "Final explanations": finals}
        df = table_map[table_choice]
        st.dataframe(df.head(500), width="stretch", hide_index=True)
        st.caption(f"Showing first {min(len(df), 500):,} of {len(df):,} rows.")


render_retrieval_tab = render_retrieval_analytics_tab
render_policy_retrieval_analytics_tab = render_retrieval_analytics_tab
