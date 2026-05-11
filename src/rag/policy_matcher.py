"""Combine Week 5 reasons with Week 6 policy retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from src.explainability.schemas import EXPLANATION_TABLE
from src.io.table_store import LocalTableStore, TableStore
from src.observability import ErrorCode, ErrorTracker
from src.rag.retriever import PolicyRetriever
from src.rag.schemas import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    FINAL_EXPLANATION_COLUMNS,
    FINAL_EXPLANATION_TABLE,
    POLICY_MATCH_COLUMNS,
    POLICY_MATCH_REPORT_FILE,
    POLICY_MATCH_TABLE,
    RAG_VERSION,
)

logger = structlog.get_logger(__name__)


def _safe_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _summarize_policy_text(text: str, *, max_chars: int = 420) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _coerce_policy_match_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=POLICY_MATCH_COLUMNS)
    out = df.copy()
    text_cols = [
        "claim_id", "risk_level", "reason_code", "reason_title", "reason_text",
        "fix_suggestion", "policy_chunk_id", "policy_text", "policy_summary",
        "source_name", "source_type", "source_path", "section_title", "retrieval_query",
        "query_policy_tags_json", "retrieved_policy_tags_json", "rag_version", "created_at",
    ]
    float_cols = ["risk_score", "similarity_score", "raw_similarity_score"]
    int_cols = ["predicted_denial", "reason_rank", "policy_rank", "page_number", "tag_overlap_count"]

    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int32")
    for col in POLICY_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[POLICY_MATCH_COLUMNS]


def _coerce_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FINAL_EXPLANATION_COLUMNS)
    out = df.copy()
    text_cols = [
        "claim_id", "risk_level", "final_explanation_text", "reasons_json",
        "policies_json", "recommended_actions_json", "rag_version", "created_at",
    ]
    float_cols = ["risk_score"]
    int_cols = ["predicted_denial"]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int32")
    for col in FINAL_EXPLANATION_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[FINAL_EXPLANATION_COLUMNS]


class PolicyMatcher:
    """Retrieve policies for Week 5 explanation reasons and write Week 6 outputs."""

    def __init__(
        self,
        *,
        gold_dir: Path,
        vector_dir: Path,
        table_store: TableStore | None = None,
        retriever: PolicyRetriever | None = None,
        error_tracker: ErrorTracker | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.gold_dir = Path(gold_dir)
        self.vector_dir = Path(vector_dir)
        self.table_store = table_store or LocalTableStore(self.gold_dir)
        self.error_tracker = error_tracker or ErrorTracker()
        self.retriever = retriever or PolicyRetriever.load(vector_dir=self.vector_dir, error_tracker=self.error_tracker)
        self.top_k = top_k
        self.min_score = min_score
        self.created_at = datetime.now(timezone.utc).isoformat()

    def _load_explanations(self, *, limit: int | None = None) -> pd.DataFrame:
        df = self.table_store.read_table(EXPLANATION_TABLE)
        if limit is not None and limit > 0:
            claim_ids = df["claim_id"].drop_duplicates().head(limit).tolist()
            df = df[df["claim_id"].isin(claim_ids)]
        return df.reset_index(drop=True)

    def _build_match_rows(self, explanation_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        match_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for _, reason in explanation_df.iterrows():
            reason_row = reason.to_dict()
            claim_id = str(reason_row.get("claim_id"))
            reason_code = str(reason_row.get("reason_code"))
            query = str(reason_row.get("policy_query") or reason_row.get("reason_text") or "")
            tags = _safe_json_list(reason_row.get("policy_tags"))
            if reason_row.get("reason_text") and reason_row.get("reason_text") not in query:
                query = f"{query}\nReason: {reason_row.get('reason_text')}"

            try:
                results = self.retriever.retrieve(query=query, policy_tags=tags, top_k=self.top_k, min_score=self.min_score)
                if not results:
                    failures.append({"claim_id": claim_id, "reason_code": reason_code, "error": "no_policy_found"})
                    continue

                for rank, result in enumerate(results, start=1):
                    meta = result.metadata
                    policy_text = str(meta.get("chunk_text") or "")
                    match_rows.append({
                        "claim_id": claim_id,
                        "risk_score": reason_row.get("risk_score"),
                        "risk_level": reason_row.get("risk_level"),
                        "predicted_denial": reason_row.get("predicted_denial"),
                        "reason_rank": reason_row.get("reason_rank"),
                        "reason_code": reason_code,
                        "reason_title": reason_row.get("reason_title"),
                        "reason_text": reason_row.get("reason_text"),
                        "fix_suggestion": reason_row.get("fix_suggestion"),
                        "policy_rank": rank,
                        "policy_chunk_id": result.chunk_id,
                        "policy_text": policy_text,
                        "policy_summary": _summarize_policy_text(policy_text),
                        "source_name": meta.get("source_name"),
                        "source_type": meta.get("source_type"),
                        "source_path": meta.get("source_path"),
                        "section_title": meta.get("section_title"),
                        "page_number": meta.get("page_number"),
                        "similarity_score": result.score,
                        "raw_similarity_score": result.raw_score,
                        "tag_overlap_count": result.tag_overlap_count,
                        "retrieval_query": query,
                        "query_policy_tags_json": json.dumps(tags),
                        "retrieved_policy_tags_json": meta.get("policy_tags_json"),
                        "rag_version": RAG_VERSION,
                        "created_at": self.created_at,
                    })
            except Exception as exc:
                self.error_tracker.record_exception(
                    exc,
                    component="rag",
                    stage="policy_match_reason",
                    fallback_code=ErrorCode.RAG_RETRIEVAL_FAILED,
                    metadata={"stage": "policy_match_reason", "claim_id": claim_id, "reason_code": reason_code},
                )
                failures.append({"claim_id": claim_id, "reason_code": reason_code, "error": str(exc)})

        return match_rows, failures

    def _build_final_rows(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return pd.DataFrame(columns=FINAL_EXPLANATION_COLUMNS)

        rows: list[dict[str, Any]] = []
        for claim_id, group in matches.sort_values(["claim_id", "reason_rank", "policy_rank"]).groupby("claim_id"):
            first = group.iloc[0].to_dict()
            reason_records: list[dict[str, Any]] = []
            policy_records: list[dict[str, Any]] = []
            actions: list[str] = []

            for reason_code, rgroup in group.groupby("reason_code", sort=False):
                reason_first = rgroup.iloc[0].to_dict()
                reason_records.append({
                    "reason_code": reason_code,
                    "reason_title": reason_first.get("reason_title"),
                    "reason_text": reason_first.get("reason_text"),
                    "fix_suggestion": reason_first.get("fix_suggestion"),
                })
                fix = str(reason_first.get("fix_suggestion") or "").strip()
                if fix and fix not in actions:
                    actions.append(fix)

                best_policy = rgroup.iloc[0].to_dict()
                policy_records.append({
                    "reason_code": reason_code,
                    "source_name": best_policy.get("source_name"),
                    "section_title": best_policy.get("section_title"),
                    "page_number": best_policy.get("page_number"),
                    "policy_summary": best_policy.get("policy_summary"),
                    "similarity_score": best_policy.get("similarity_score"),
                })

            reason_lines = [f"- {item['reason_title']}: {item['reason_text']}" for item in reason_records]
            policy_lines = [
                f"- {item['source_name']}"
                + (f" / {item['section_title']}" if item.get("section_title") else "")
                + f": {item['policy_summary']}"
                for item in policy_records
            ]
            action_lines = [f"- {action}" for action in actions]
            final_text = (
                f"Claim {claim_id} is {first.get('risk_level')} risk with score {float(first.get('risk_score') or 0):.2%}.\n\n"
                "Reasons:\n" + "\n".join(reason_lines[:3]) + "\n\n"
                "Policy support:\n" + "\n".join(policy_lines[:3]) + "\n\n"
                "Recommended actions:\n" + "\n".join(action_lines[:3])
            )

            rows.append({
                "claim_id": claim_id,
                "risk_score": first.get("risk_score"),
                "risk_level": first.get("risk_level"),
                "predicted_denial": first.get("predicted_denial"),
                "final_explanation_text": final_text,
                "reasons_json": json.dumps(reason_records, default=str),
                "policies_json": json.dumps(policy_records, default=str),
                "recommended_actions_json": json.dumps(actions, default=str),
                "rag_version": RAG_VERSION,
                "created_at": self.created_at,
            })

        return pd.DataFrame(rows, columns=FINAL_EXPLANATION_COLUMNS)

    def run(self, *, limit: int | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {"status": "started", "component": "rag", "created_at": self.created_at}
        try:
            explanations = self._load_explanations(limit=limit)
            rows, failures = self._build_match_rows(explanations)
            match_df = _coerce_policy_match_schema(pd.DataFrame(rows))
            final_df = _coerce_final_schema(self._build_final_rows(match_df))

            match_path = self.table_store.write_table(POLICY_MATCH_TABLE, match_df)
            final_path = self.table_store.write_table(FINAL_EXPLANATION_TABLE, final_df)

            if failures:
                self.error_tracker.record(
                    ErrorCode.RAG_NO_RELEVANT_POLICY_FOUND,
                    f"{len(failures)} reason(s) did not receive policy evidence.",
                    component="rag",
                    stage="policy_match_batch",
                    metadata={"stage": "policy_match_batch", "failure_count": len(failures)},
                )

            report.update({
                "status": "success_with_warnings" if failures else "success",
                "reason_rows_input": int(len(explanations)),
                "policy_match_rows": int(len(match_df)),
                "final_explanation_rows": int(len(final_df)),
                "unmatched_reason_count": int(len(failures)),
                "unmatched_reasons": failures[:20],
                "policy_match_path": str(match_path),
                "final_explanation_path": str(final_path),
            })

            report_path = self.gold_dir / POLICY_MATCH_REPORT_FILE
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(report_path)
            logger.info("policy_matching_complete", **report)
            return report
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="rag",
                stage="policy_match_run",
                fallback_code=ErrorCode.RAG_UNEXPECTED,
                metadata={"stage": "policy_match_run"},
            )
            logger.exception("policy_matching_failed", error=str(exc))
            report.update({"status": "failed", "error": str(exc)})
            return report
