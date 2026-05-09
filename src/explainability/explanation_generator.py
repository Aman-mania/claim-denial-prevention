"""
Week 5 Explainability Pipeline.

Inputs:
- data/gold/gold_claim_features.parquet
- models/xgb_model.pkl + training_report.json

Outputs:
- data/gold/gold_claim_explanations.parquet
- data/gold/gold_claim_explanation_summary.parquet
- data/gold/explanation_report.json

Cloud-readiness:
The business logic reads/writes through TableStore boundaries. Local execution
uses Parquet; Databricks migration can replace the store with Delta/Unity
Catalog without changing the reason-mapping code.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math

import pandas as pd
import structlog

from src.explainability.reason_mapper import ReasonMapper
from src.explainability.schemas import (
    EXPLANATION_COLUMNS,
    EXPLANATION_REPORT_FILE,
    EXPLANATION_SUMMARY_TABLE,
    EXPLANATION_TABLE,
    EXPLANATION_VERSION,
    GOLD_FEATURE_TABLE,
    SUMMARY_COLUMNS,
)
from src.io.table_store import LocalTableStore, TableStore
from src.observability import ClaimDenialError, ErrorCode, ErrorTracker

logger = structlog.get_logger(__name__)


def _serialize_feature_value(value: Any) -> str | None:
    """Return a Parquet-safe text representation of a raw feature value.

    Explanation rows intentionally store values from many source features in a
    single column. Some features are booleans, some are floats, some are strings.
    PyArrow/Parquet requires one physical type per column, so we serialize this
    mixed-value evidence field as text while keeping SHAP/risk fields numeric.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return str(value)

    try:
        return json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        return str(value)


def _coerce_explanation_long_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce explanation output to stable Parquet-compatible column types."""
    if df.empty:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)

    out = df.copy()
    text_cols = [
        "claim_id", "risk_level", "reason_code", "reason_title", "reason_text",
        "business_category", "evidence_type", "feature_name", "feature_label",
        "feature_value", "shap_direction", "shap_output_unit", "fix_suggestion",
        "policy_query", "policy_tags", "model_used", "explanation_version", "created_at",
    ]
    float_cols = ["risk_score", "classification_threshold", "review_threshold", "shap_value"]
    int_cols = ["predicted_denial", "reason_rank"]

    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].map(_serialize_feature_value).astype("string")
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out[EXPLANATION_COLUMNS]


def _coerce_explanation_summary_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce summary output to stable Parquet-compatible column types."""
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    out = df.copy()
    text_cols = [
        "claim_id", "risk_level", "reason_1", "reason_2", "reason_3",
        "reason_codes", "reason_texts_json", "fix_suggestions_json",
        "policy_queries_json", "policy_tags_json", "model_used",
        "explanation_version", "created_at",
    ]
    float_cols = ["risk_score", "classification_threshold", "review_threshold"]
    int_cols = ["predicted_denial"]

    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].map(_serialize_feature_value).astype("string")
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out[SUMMARY_COLUMNS]


class ExplanationGenerationPipeline:
    """Generate business explanations for Gold claim predictions."""

    def __init__(
        self,
        *,
        gold_dir: Path,
        models_dir: Path,
        output_dir: Path | None = None,
        input_store: TableStore | None = None,
        table_store: TableStore | None = None,
        error_tracker: ErrorTracker | None = None,
        max_reasons: int = 3,
        shap_top_n: int = 10,
        model_name: str = "xgboost",
    ) -> None:
        self.gold_dir = Path(gold_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir or gold_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_store = input_store or LocalTableStore(self.gold_dir)
        self.table_store = table_store or LocalTableStore(self.output_dir)
        self.error_tracker = error_tracker or ErrorTracker()
        self.max_reasons = max_reasons
        self.shap_top_n = max(shap_top_n, max_reasons)
        self.model_name = model_name
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.reason_mapper = ReasonMapper(max_reasons=max_reasons)

    def _load_gold_features(self) -> pd.DataFrame:
        try:
            df = self.input_store.read_table(GOLD_FEATURE_TABLE)
            logger.info("gold_features_loaded_for_explainability", rows=len(df), cols=len(df.columns))
            return df
        except Exception as exc:
            self.error_tracker.record_exception(
                exc,
                component="xai",
                stage="load_gold_features",
                fallback_code=ErrorCode.XAI_GOLD_FEATURES_MISSING,
                metadata={"stage": "load_gold_features", "table": GOLD_FEATURE_TABLE},
            )
            raise

    def _load_services(self):
        from src.ml.predict import ClaimPredictor
        from src.ml.explain import SHAPExplainer

        if self.model_name != "xgboost":
            event = self.error_tracker.record(
                ErrorCode.XAI_MODEL_EXPLAINER_MISMATCH,
                "Week 5 currently explains XGBoost predictions only, because SHAPExplainer uses TreeExplainer.",
                component="xai",
                stage="load_services",
                metadata={"stage": "load_services", "model_name": self.model_name},
            )
            raise RuntimeError(event.message)

        predictor = ClaimPredictor.load(models_dir=self.models_dir, model_name="xgboost")
        xgb_path = self.models_dir / "xgb_model.pkl"
        if not xgb_path.exists():
            event = self.error_tracker.record(
                ErrorCode.XAI_SHAP_EXPLAINER_MISSING,
                f"XGBoost model not found for SHAP explanations: {xgb_path}",
                component="xai",
                stage="load_services",
                metadata={"stage": "load_services", "path": str(xgb_path)},
            )
            raise FileNotFoundError(event.message)
        explainer = SHAPExplainer.from_model_file(xgb_path)
        logger.info("xai_services_loaded", model="xgboost", shap_model_path=str(xgb_path))
        return predictor, explainer

    def _filter_claims(
        self,
        df: pd.DataFrame,
        *,
        limit: int | None = None,
        claim_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        out = df
        if claim_ids:
            claim_set = {str(c) for c in claim_ids}
            out = out[out["claim_id"].astype(str).isin(claim_set)]
        if limit is not None and limit > 0:
            out = out.head(limit)
        return out.reset_index(drop=True)

    def _build_long_rows(
        self,
        *,
        claim: dict[str, Any],
        prediction: dict[str, Any],
        mapped_reasons: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for reason in mapped_reasons:
            row = {
                "claim_id": claim.get("claim_id"),
                "risk_score": prediction.get("risk_score"),
                "risk_level": prediction.get("risk_level"),
                "predicted_denial": prediction.get("predicted_denial"),
                "classification_threshold": prediction.get("classification_threshold"),
                "review_threshold": prediction.get("review_threshold"),
                "model_used": prediction.get("model_used"),
                "explanation_version": EXPLANATION_VERSION,
                "created_at": self.created_at,
                **{k: v for k, v in reason.items() if k != "reason_definition"},
            }
            if isinstance(row.get("policy_tags"), list):
                row["policy_tags"] = json.dumps(row["policy_tags"])
            row["feature_value"] = _serialize_feature_value(row.get("feature_value"))
            rows.append(row)
        return rows

    def _build_summary(self, long_df: pd.DataFrame) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)

        summary_rows: list[dict[str, Any]] = []
        for claim_id, group in long_df.sort_values(["claim_id", "reason_rank"]).groupby("claim_id"):
            first = group.iloc[0].to_dict()
            reasons = group["reason_text"].tolist()
            fixes = group["fix_suggestion"].tolist()
            policy_queries = group["policy_query"].tolist()
            policy_tags: list[str] = []
            for raw_tags in group["policy_tags"].tolist():
                try:
                    tags = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
                except Exception:
                    tags = []
                for tag in tags or []:
                    if tag not in policy_tags:
                        policy_tags.append(tag)

            summary_rows.append({
                "claim_id": claim_id,
                "risk_score": first.get("risk_score"),
                "risk_level": first.get("risk_level"),
                "predicted_denial": first.get("predicted_denial"),
                "classification_threshold": first.get("classification_threshold"),
                "review_threshold": first.get("review_threshold"),
                "reason_1": reasons[0] if len(reasons) > 0 else None,
                "reason_2": reasons[1] if len(reasons) > 1 else None,
                "reason_3": reasons[2] if len(reasons) > 2 else None,
                "reason_codes": json.dumps(group["reason_code"].tolist()),
                "reason_texts_json": json.dumps(reasons),
                "fix_suggestions_json": json.dumps(fixes),
                "policy_queries_json": json.dumps(policy_queries),
                "policy_tags_json": json.dumps(policy_tags),
                "model_used": first.get("model_used"),
                "explanation_version": EXPLANATION_VERSION,
                "created_at": first.get("created_at"),
            })

        return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    def _normalize_long_df(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        long_df = pd.DataFrame(rows)
        if long_df.empty:
            return pd.DataFrame(columns=EXPLANATION_COLUMNS)
        for col in EXPLANATION_COLUMNS:
            if col not in long_df.columns:
                long_df[col] = None
        return _coerce_explanation_long_schema(long_df)

    def _write_outputs(self, long_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[Path | str, Path | str]:
        try:
            long_df = _coerce_explanation_long_schema(long_df)
            summary_df = _coerce_explanation_summary_schema(summary_df)
            long_path = self.table_store.write_table(EXPLANATION_TABLE, long_df)
            summary_path = self.table_store.write_table(EXPLANATION_SUMMARY_TABLE, summary_df)
            return long_path, summary_path
        except Exception as exc:
            raise ClaimDenialError(
                ErrorCode.XAI_EXPLANATION_WRITE_FAILED,
                str(exc),
                component="xai",
                metadata={"stage": "write_explanation_outputs"},
            ) from exc

    def run(
        self,
        *,
        limit: int | None = None,
        claim_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "started",
            "created_at": self.created_at,
            "explanation_version": EXPLANATION_VERSION,
            "component": "xai",
        }

        try:
            df = self._filter_claims(self._load_gold_features(), limit=limit, claim_ids=claim_ids)
            predictor, explainer = self._load_services()

            long_rows: list[dict[str, Any]] = []
            failed_claims: list[dict[str, str]] = []
            unmapped_features: dict[str, int] = {}

            for _, row in df.iterrows():
                claim = row.to_dict()
                claim_id = str(claim.get("claim_id"))
                try:
                    prediction = predictor.predict(claim)
                    shap_explanation = explainer.explain(claim, top_n=self.shap_top_n)
                    mapped = self.reason_mapper.map(
                        shap_explanation=shap_explanation,
                        claim_features=claim,
                        prediction=prediction,
                    )
                    for feature in self.reason_mapper.last_unmapped_features:
                        unmapped_features[feature] = unmapped_features.get(feature, 0) + 1
                    if not mapped:
                        self.error_tracker.record(
                            ErrorCode.XAI_NO_REASON_GENERATED,
                            "No business reasons generated for a claim.",
                            component="xai",
                            stage="claim_explanation",
                            metadata={"stage": "claim_explanation", "claim_id": claim_id},
                        )
                    long_rows.extend(
                        self._build_long_rows(
                            claim=claim,
                            prediction=prediction,
                            mapped_reasons=mapped,
                        )
                    )
                except Exception as exc:
                    logger.exception("claim_explanation_failed", claim_id=claim_id, error=str(exc))
                    event = self.error_tracker.record_exception(
                        exc,
                        component="xai",
                        stage="claim_explanation",
                        fallback_code=ErrorCode.XAI_EXPLANATION_GENERATION_FAILED,
                        metadata={"stage": "claim_explanation", "claim_id": claim_id},
                    )
                    failed_claims.append({"claim_id": claim_id, "error": str(exc), "error_code": event.error_code})

            long_df = self._normalize_long_df(long_rows)
            summary_df = _coerce_explanation_summary_schema(self._build_summary(long_df))
            long_path, summary_path = self._write_outputs(long_df, summary_df)

            if failed_claims:
                self.error_tracker.record(
                    ErrorCode.XAI_EXPLANATION_PARTIAL_FAILURE,
                    f"{len(failed_claims)} claim explanation(s) failed during batch generation.",
                    component="xai",
                    stage="batch_explanation",
                    metadata={"stage": "batch_explanation", "failure_count": len(failed_claims)},
                )

            risk_counts = summary_df["risk_level"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
            reason_counts = long_df["reason_code"].value_counts(dropna=False).head(20).to_dict() if not long_df.empty else {}

            status = "success_with_warnings" if failed_claims else "success"
            report.update({
                "status": status,
                "claims_input": int(len(df)),
                "claims_explained": int(summary_df["claim_id"].nunique()) if not summary_df.empty else 0,
                "reason_rows": int(len(long_df)),
                "failed_claims": failed_claims,
                "failed_claim_count": len(failed_claims),
                "unmapped_features": unmapped_features,
                "risk_level_counts": risk_counts,
                "top_reason_counts": reason_counts,
                "explanations_path": str(long_path),
                "summary_path": str(summary_path),
            })

            report_path = self.output_dir / EXPLANATION_REPORT_FILE
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            report["report_path"] = str(report_path)

            logger.info("explanation_generation_complete", **report)
        except Exception as exc:
            logger.exception("explanation_generation_failed", error=str(exc))
            err_stage = "run_explain"
            fallback_code = ErrorCode.XAI_UNEXPECTED
            metadata = {"stage": "run_explain"}
            if isinstance(exc, ClaimDenialError):
                err_stage = str(exc.metadata.get("stage", err_stage))
                metadata = {**exc.metadata, "stage": err_stage}
                fallback_code = exc.code
            event = self.error_tracker.record_exception(
                exc,
                component="xai",
                stage=err_stage,
                fallback_code=fallback_code,
                metadata=metadata,
            )
            report.update({
                "status": "failed",
                "error": str(exc),
                "error_code": event.error_code,
                "error_event_id": event.event_id,
            })

        return report
