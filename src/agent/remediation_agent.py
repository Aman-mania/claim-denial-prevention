"""Week 7 integrated remediation agent.

The agent composes existing project layers without replacing them:
1. lightweight input validation;
2. Week 4 custom-claim scoring service;
3. Week 5 SHAP + business reason mapping;
4. Week 6 RAG policy retrieval;
5. deterministic recommendation catalog;
6. optional OpenAI presentation layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.agent.openai_output import OpenAIPresentationLayer, deterministic_presentation
from src.agent.recommendation_catalog import RecommendationCatalog
from src.rules.claim_validator import ClaimInputValidator

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
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(v) for v in decoded]
        except Exception:
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _summarize_policy_text(text: str, max_chars: int = 420) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class RemediationAgent:
    """End-to-end local Week 7 agent for one custom claim."""

    def __init__(
        self,
        *,
        validator: ClaimInputValidator,
        claim_service: Any,
        shap_explainer: Any | None,
        reason_mapper: Any | None,
        retriever: Any | None,
        recommendation_catalog: RecommendationCatalog | None = None,
        presentation_layer: OpenAIPresentationLayer | None = None,
        top_k_policy: int = 3,
        min_policy_score: float = 0.20,
    ) -> None:
        self.validator = validator
        self.claim_service = claim_service
        self.shap_explainer = shap_explainer
        self.reason_mapper = reason_mapper
        self.retriever = retriever
        self.recommendation_catalog = recommendation_catalog or RecommendationCatalog()
        self.presentation_layer = presentation_layer or OpenAIPresentationLayer()
        self.top_k_policy = int(top_k_policy)
        self.min_policy_score = float(min_policy_score)

    @classmethod
    def load(
        cls,
        *,
        gold_dir: Path,
        models_dir: Path,
        vector_dir: Path,
        enable_openai: bool | None = None,
        top_k_policy: int = 3,
        min_policy_score: float = 0.20,
    ) -> "RemediationAgent":
        """Load all local artifacts needed by the integrated agent."""
        from src.inference.claim_service import ClaimDenialService

        validator = ClaimInputValidator.from_gold_dir(gold_dir)
        claim_service = ClaimDenialService.load(gold_dir=gold_dir, models_dir=models_dir)

        shap_explainer = None
        reason_mapper = None
        xgb_path = Path(models_dir) / "xgb_model.pkl"
        if xgb_path.exists():
            try:
                from src.ml.explain import SHAPExplainer
                from src.explainability.reason_mapper import ReasonMapper

                shap_explainer = SHAPExplainer.from_model_file(xgb_path)
                reason_mapper = ReasonMapper(max_reasons=3)
            except Exception as exc:
                logger.warning("agent_xai_load_failed", error=str(exc))

        retriever = None
        metadata_path = Path(vector_dir) / "policy_metadata.json"
        if metadata_path.exists():
            try:
                from src.rag.retriever import PolicyRetriever

                retriever = PolicyRetriever.load(vector_dir=vector_dir)
            except Exception as exc:
                logger.warning("agent_rag_load_failed", error=str(exc))

        presentation = OpenAIPresentationLayer(enabled=enable_openai)
        return cls(
            validator=validator,
            claim_service=claim_service,
            shap_explainer=shap_explainer,
            reason_mapper=reason_mapper,
            retriever=retriever,
            presentation_layer=presentation,
            top_k_policy=top_k_policy,
            min_policy_score=min_policy_score,
        )

    def _should_generate_risk_reasons(self, prediction: dict[str, Any]) -> bool:
        """Return True only when model risk needs explanation/remediation.

        LOW-risk claims should not receive denial-risk reasons just because tiny
        positive SHAP values exist. Those small values are useful for technical
        debugging, but they are confusing in the business analyst experience.
        Validation warnings still remain visible separately.
        """
        risk_level = str(prediction.get("risk_level", "LOW")).upper()
        return risk_level in {"HIGH", "MEDIUM"}

    def _build_reasons(self, *, features: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._should_generate_risk_reasons(prediction):
            logger.info(
                "agent_low_risk_reasons_suppressed",
                claim_id=features.get("claim_id"),
                risk_level=prediction.get("risk_level"),
                risk_score=prediction.get("risk_score"),
            )
            return []

        if not self.shap_explainer or not self.reason_mapper:
            return self._fallback_reasons(features=features, prediction=prediction)
        try:
            shap_explanation = self.shap_explainer.explain(features, top_n=10)
            reasons = self.reason_mapper.map(
                shap_explanation=shap_explanation,
                claim_features=features,
                prediction=prediction,
            )
            for rank, reason in enumerate(reasons, start=1):
                reason.setdefault("reason_rank", rank)
            return reasons
        except Exception as exc:
            logger.warning("agent_reason_generation_failed_using_fallback", error=str(exc))
            return self._fallback_reasons(features=features, prediction=prediction)

    def _fallback_reasons(self, *, features: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
        reason_rows: list[dict[str, Any]] = []
        candidates = [
            ("diagnosis_code_missing", "MISSING_DIAGNOSIS", "Diagnosis is missing", "The claim is missing diagnosis support.", "Add or verify ICD diagnosis code before submission.", ["diagnosis", "medical_necessity"]),
            ("procedure_code_missing", "MISSING_PROCEDURE", "Procedure is missing", "The claim is missing a billed procedure/service code.", "Add the billed procedure code before submission.", ["procedure_coding", "claim_completeness"]),
            ("proc_no_diag", "PROCEDURE_WITHOUT_DIAGNOSIS", "Procedure without diagnosis", "A procedure is present without a supporting diagnosis.", "Link the procedure to a supporting diagnosis code.", ["diagnosis", "medical_necessity"]),
            ("is_high_cost", "HIGH_COST_CLAIM", "High-cost claim", "The billed amount is high compared with benchmark context.", "Check authorization/documentation requirements for high-cost claims.", ["high_cost", "documentation"]),
        ]
        for feature, code, title, text, fix, tags in candidates:
            if features.get(feature) in {True, 1, "1", "true", "True"}:
                reason_rows.append({
                    "reason_rank": len(reason_rows) + 1,
                    "reason_code": code,
                    "reason_title": title,
                    "reason_text": text,
                    "feature_name": feature,
                    "feature_value": features.get(feature),
                    "evidence_type": "agent_fallback_rule",
                    "fix_suggestion": fix,
                    "policy_query": text,
                    "policy_tags": tags,
                    "shap_value": 0.0,
                    "shap_direction": "unknown",
                })
            if len(reason_rows) >= 3:
                break
        if not reason_rows and str(prediction.get("risk_level", "LOW")).upper() in {"HIGH", "MEDIUM"}:
            reason_rows.append({
                "reason_rank": 1,
                "reason_code": "MODEL_REVIEW_RISK",
                "reason_title": "Model review risk",
                "reason_text": "The model indicates this claim should be reviewed before submission.",
                "feature_name": None,
                "feature_value": None,
                "evidence_type": "agent_fallback_model",
                "fix_suggestion": "Review claim details and supporting documentation before submission.",
                "policy_query": "claim review required when risk is elevated",
                "policy_tags": ["documentation", "claim_completeness"],
                "shap_value": 0.0,
                "shap_direction": "unknown",
            })
        return reason_rows

    def _retrieve_policy_evidence(self, reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.retriever:
            return []
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for reason in reasons:
            query = str(reason.get("policy_query") or reason.get("reason_text") or "").strip()
            if reason.get("reason_text") and str(reason.get("reason_text")) not in query:
                query = f"{query}\nReason: {reason.get('reason_text')}"
            if not query:
                continue
            tags = _safe_json_list(reason.get("policy_tags"))
            try:
                results = self.retriever.retrieve(
                    query=query,
                    policy_tags=tags,
                    top_k=self.top_k_policy,
                    min_score=self.min_policy_score,
                )
            except Exception as exc:
                logger.warning("agent_policy_retrieval_failed", reason_code=reason.get("reason_code"), error=str(exc))
                continue
            for result in results:
                meta = result.metadata
                key = (str(reason.get("reason_code")), str(result.chunk_id))
                if key in seen:
                    continue
                seen.add(key)
                policy_text = str(meta.get("chunk_text") or "")
                evidence.append({
                    "reason_code": reason.get("reason_code"),
                    "policy_chunk_id": result.chunk_id,
                    "source_name": meta.get("source_name"),
                    "source_type": meta.get("source_type"),
                    "source_path": meta.get("source_path"),
                    "section_title": meta.get("section_title"),
                    "page_number": meta.get("page_number"),
                    "policy_summary": _summarize_policy_text(policy_text),
                    "similarity_score": round(float(result.score), 4),
                    "raw_similarity_score": round(float(result.raw_score), 4),
                    "tag_overlap_count": int(result.tag_overlap_count),
                })
        return evidence

    def analyze_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Run the full Week 7 agent flow for one custom claim."""
        validation = self.validator.validate(claim)
        validation_dict = validation.to_dict()
        claim_id = validation.normalized_claim.get("claim_id") or (claim.get("claim_id") if isinstance(claim, dict) else None)

        if not validation.is_valid:
            payload = {
                "status": "blocked",
                "claim_id": claim_id,
                "validation": validation_dict,
                "prediction": None,
                "features": None,
                "reasons": [],
                "policy_evidence": [],
                "recommendations": [],
                "decision": self.recommendation_catalog.decision(validation=validation_dict, prediction=None, recommendations=[]),
            }
            payload["recommendations"] = self.recommendation_catalog.generate(
                validation=validation_dict,
                prediction=None,
                reasons=[],
                policy_evidence=[],
            )
            payload["agent_presentation"] = self.presentation_layer.generate(payload)
            return payload

        scored = self.claim_service.score_claim(validation.normalized_claim)
        if scored.get("status") != "success":
            payload = {
                "status": "error",
                "claim_id": claim_id,
                "validation": validation_dict,
                "prediction": None,
                "features": None,
                "reasons": [],
                "policy_evidence": [],
                "recommendations": [],
                "decision": {"status": "BLOCKED", "priority": "HIGH", "summary": "Inference failed. Review error details before retrying."},
                "error": scored.get("error"),
            }
            payload["agent_presentation"] = deterministic_presentation(payload, source="deterministic_error")
            return payload

        prediction = scored.get("prediction") or {}
        features = scored.get("features") or {}
        reasons = self._build_reasons(features=features, prediction=prediction)
        evidence = self._retrieve_policy_evidence(reasons)
        recommendations = self.recommendation_catalog.generate(
            validation=validation_dict,
            prediction=prediction,
            reasons=reasons,
            policy_evidence=evidence,
        )
        decision = self.recommendation_catalog.decision(
            validation=validation_dict,
            prediction=prediction,
            recommendations=recommendations,
        )
        payload = {
            "status": "success",
            "claim_id": claim_id,
            "validation": validation_dict,
            "prediction": prediction,
            "features": features,
            "reasons": reasons,
            "policy_evidence": evidence,
            "recommendations": recommendations,
            "decision": decision,
        }
        payload["agent_presentation"] = self.presentation_layer.generate(payload)
        logger.info("agent_claim_analyzed", claim_id=claim_id, risk_level=prediction.get("risk_level"), decision=decision.get("status"))
        return payload

    # Alias used by future FastAPI route naming.
    recommend = analyze_claim
