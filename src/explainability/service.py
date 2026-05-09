"""On-demand explainability service for custom claims and future FastAPI.

Batch mode (`run_explain.py`) materializes explanation tables for all Gold
claims. This service supports dynamic explanations for claims entered later in
Streamlit/FastAPI without regenerating the whole batch table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.explainability.reason_mapper import ReasonMapper
from src.observability import ErrorCode, ErrorTracker

logger = structlog.get_logger(__name__)


class ExplanationService:
    """Generate prediction + SHAP + business reasons for one claim."""

    def __init__(self, *, predictor: Any, explainer: Any, reason_mapper: ReasonMapper, feature_builder: Any | None = None, error_tracker: ErrorTracker | None = None) -> None:
        self.predictor = predictor
        self.explainer = explainer
        self.reason_mapper = reason_mapper
        self.feature_builder = feature_builder
        self.error_tracker = error_tracker or ErrorTracker()

    @classmethod
    def load(
        cls,
        *,
        gold_dir: Path,
        models_dir: Path,
        model_name: str = "xgboost",
        max_reasons: int = 3,
        error_tracker: ErrorTracker | None = None,
    ) -> "ExplanationService":
        tracker = error_tracker or ErrorTracker()
        try:
            from src.ml.predict import ClaimPredictor
            from src.ml.explain import SHAPExplainer
            from src.inference.feature_builder import CustomClaimFeatureBuilder

            predictor = ClaimPredictor.load(models_dir=Path(models_dir), model_name=model_name)
            model_path = Path(models_dir) / "xgb_model.pkl"
            if model_name != "xgboost":
                event = tracker.record(
                    ErrorCode.XAI_MODEL_EXPLAINER_MISMATCH,
                    f"Week 5 SHAP explanations currently require XGBoost; got model_name={model_name}",
                    component="xai",
                    stage="service_load",
                    metadata={"stage": "service_load", "model_name": model_name},
                )
                raise RuntimeError(f"{event.error_code}: XAI currently supports xgboost explanations only.")
            if not model_path.exists():
                event = tracker.record(
                    ErrorCode.XAI_SHAP_EXPLAINER_MISSING,
                    f"XGBoost model file missing: {model_path}",
                    component="xai",
                    stage="service_load",
                    metadata={"stage": "service_load", "path": str(model_path)},
                )
                raise FileNotFoundError(event.message)

            explainer = SHAPExplainer.from_model_file(model_path)
            try:
                feature_builder = CustomClaimFeatureBuilder.load(gold_dir=Path(gold_dir))
            except Exception:
                # Feature builder is only required for raw custom claims. Feature-row
                # explanations can still work without it.
                logger.warning("custom_claim_feature_builder_not_loaded", gold_dir=str(gold_dir))
                feature_builder = None

            return cls(
                predictor=predictor,
                explainer=explainer,
                reason_mapper=ReasonMapper(max_reasons=max_reasons),
                feature_builder=feature_builder,
                error_tracker=tracker,
            )
        except Exception as exc:
            tracker.record_exception(
                exc,
                component="xai",
                stage="service_load",
                fallback_code=ErrorCode.XAI_UNEXPECTED,
                metadata={"stage": "service_load"},
            )
            raise

    def explain_feature_row(self, claim_features: dict[str, Any], *, shap_top_n: int = 10) -> dict[str, Any]:
        """Explain one already model-ready feature row."""
        try:
            prediction = self.predictor.predict(claim_features)
            shap_explanation = self.explainer.explain(claim_features, top_n=shap_top_n)
            reasons = self.reason_mapper.map(
                shap_explanation=shap_explanation,
                claim_features=claim_features,
                prediction=prediction,
            )
            if not reasons:
                self.error_tracker.record(
                    ErrorCode.XAI_NO_REASON_GENERATED,
                    "No business reason could be mapped for the claim explanation.",
                    component="xai",
                    stage="explain_feature_row",
                    metadata={"stage": "explain_feature_row", "claim_id": claim_features.get("claim_id")},
                )
            return {
                "status": "success",
                "claim_id": claim_features.get("claim_id"),
                "prediction": prediction,
                "reasons": reasons,
                "shap": shap_explanation,
                "errors": [],
            }
        except Exception as exc:
            event = self.error_tracker.record_exception(
                exc,
                component="xai",
                stage="explain_feature_row",
                fallback_code=ErrorCode.XAI_EXPLANATION_GENERATION_FAILED,
                metadata={"stage": "explain_feature_row", "claim_id": claim_features.get("claim_id")},
            )
            return {"status": "error", "claim_id": claim_features.get("claim_id"), "error": event.__dict__, "prediction": None, "reasons": []}

    def explain_raw_claim(self, raw_claim: dict[str, Any], *, shap_top_n: int = 10) -> dict[str, Any]:
        """Build features for a raw custom claim and explain it on demand."""
        if self.feature_builder is None:
            event = self.error_tracker.record(
                ErrorCode.XAI_FEATURE_BUILD_FAILED,
                "Feature builder is unavailable. Run run_gold.py before explaining raw custom claims.",
                component="xai",
                stage="explain_raw_claim",
                metadata={"stage": "explain_raw_claim", "claim_id": raw_claim.get("claim_id")},
            )
            return {"status": "error", "claim_id": raw_claim.get("claim_id"), "error": event.__dict__, "prediction": None, "reasons": []}

        try:
            try:
                features = self.feature_builder.build_features(raw_claim)
            except AttributeError:
                features = self.feature_builder.build(raw_claim)
            result = self.explain_feature_row(features, shap_top_n=shap_top_n)
            result["features"] = features
            return result
        except Exception as exc:
            event = self.error_tracker.record_exception(
                exc,
                component="xai",
                stage="explain_raw_claim",
                fallback_code=ErrorCode.XAI_FEATURE_BUILD_FAILED,
                metadata={"stage": "explain_raw_claim", "claim_id": raw_claim.get("claim_id")},
            )
            return {"status": "error", "claim_id": raw_claim.get("claim_id"), "error": event.__dict__, "prediction": None, "reasons": []}
