"""
Safe inference service for the Custom Claim Builder.

This is the production-facing wrapper the dashboard/API should call instead of
manually constructing ML feature dictionaries. It centralizes:
- raw claim validation
- feature building
- model prediction
- error-code responses
- repeated error tracking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker

logger = structlog.get_logger(__name__)


class ClaimDenialService:
    """End-to-end custom claim scoring service with structured error handling."""

    def __init__(self, feature_builder: Any, predictor: Any, error_tracker: ErrorTracker | None = None) -> None:
        self.feature_builder = feature_builder
        self.predictor = predictor
        self.error_tracker = error_tracker or ErrorTracker()

    @classmethod
    def load(
        cls,
        *,
        gold_dir: Path,
        models_dir: Path,
        error_tracker: ErrorTracker | None = None,
    ) -> "ClaimDenialService":
        """Load feature builder + recommended predictor."""
        tracker = error_tracker or ErrorTracker()
        try:
            from src.inference.feature_builder import ClaimFeatureBuilder
        except Exception as exc:
            tracker.record(
                ErrorCode.INFER_ARTIFACT_NOT_FOUND,
                "Could not import ClaimFeatureBuilder. Apply the inference feature-builder patch first.",
                component="inference",
                stage="service_load",
                metadata={"stage": "service_load"},
                exc=exc,
            )
            raise

        try:
            from src.ml.predict import ClaimPredictor
        except Exception as exc:
            tracker.record(
                ErrorCode.INFER_MODEL_NOT_FOUND,
                "Could not import ClaimPredictor. Train the model and check src/ml/predict.py.",
                component="inference",
                stage="service_load",
                metadata={"stage": "service_load"},
                exc=exc,
            )
            raise

        try:
            feature_builder = ClaimFeatureBuilder.load(gold_dir=gold_dir)
        except Exception as exc:
            tracker.record_exception(
                exc,
                component="inference",
                stage="load_feature_builder",
                fallback_code=ErrorCode.INFER_ARTIFACT_NOT_FOUND,
                metadata={"stage": "load_feature_builder", "path": str(gold_dir)},
            )
            raise

        try:
            predictor = ClaimPredictor.recommended(models_dir=models_dir)
        except Exception as exc:
            tracker.record_exception(
                exc,
                component="inference",
                stage="load_predictor",
                fallback_code=ErrorCode.INFER_MODEL_NOT_FOUND,
                metadata={"stage": "load_predictor", "path": str(models_dir)},
            )
            raise

        return cls(feature_builder=feature_builder, predictor=predictor, error_tracker=tracker)

    def _validate_raw_claim(self, claim: dict[str, Any]) -> None:
        """Validate raw-claim-level input, not derived ML features."""
        if not isinstance(claim, dict):
            raise ClaimDenialError(
                ErrorCode.INFER_INVALID_CLAIM,
                "Custom claim must be a dictionary/object.",
                component="inference",
                metadata={"stage": "validate_raw_claim"},
            )

        required = ["claim_id", "patient_id", "provider_id"]
        missing = [field for field in required if not claim.get(field)]
        if missing:
            raise ClaimDenialError(
                ErrorCode.INFER_INVALID_CLAIM,
                f"Missing required claim fields: {missing}",
                component="inference",
                metadata={"stage": "validate_raw_claim", "field": ",".join(missing)},
            )

        # Guard against the old dashboard sending contradictory feature-level data.
        if claim.get("billed_amount_missing") is True and claim.get("billed_amount") is not None:
            raise ClaimDenialError(
                ErrorCode.INFER_INCONSISTENT_CLAIM_STATE,
                "Inconsistent claim: billed_amount_missing=True while billed_amount is present. "
                "Clear billed_amount or send only raw claim fields.",
                component="inference",
                metadata={"stage": "validate_raw_claim", "field": "billed_amount"},
            )
        if claim.get("diagnosis_code_missing") is True and claim.get("diagnosis_code") not in {None, "", "MISSING"}:
            raise ClaimDenialError(
                ErrorCode.INFER_INCONSISTENT_CLAIM_STATE,
                "Inconsistent claim: diagnosis_code_missing=True while diagnosis_code is present.",
                component="inference",
                metadata={"stage": "validate_raw_claim", "field": "diagnosis_code"},
            )
        if claim.get("procedure_code_missing") is True and claim.get("procedure_code") not in {None, "", "MISSING"}:
            raise ClaimDenialError(
                ErrorCode.INFER_INCONSISTENT_CLAIM_STATE,
                "Inconsistent claim: procedure_code_missing=True while procedure_code is present.",
                component="inference",
                metadata={"stage": "validate_raw_claim", "field": "procedure_code"},
            )

    def score_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """
        Score one raw claim.

        Returns a success/error envelope so dashboards and FastAPI can display
        stable error codes instead of crashing.
        """
        try:
            self._validate_raw_claim(claim)

            try:
                features = self.feature_builder.build_features(claim)
            except AttributeError:
                # Backward-compatible method name used in some prior patches.
                features = self.feature_builder.build(claim)
            except Exception as exc:
                event = self.error_tracker.record_exception(
                    exc,
                    component="inference",
                    stage="feature_build",
                    fallback_code=ErrorCode.INFER_FEATURE_BUILD_FAILED,
                    metadata={"stage": "feature_build"},
                )
                return self._error_response(event)

            try:
                prediction = self.predictor.predict(features)
            except Exception as exc:
                event = self.error_tracker.record_exception(
                    exc,
                    component="inference",
                    stage="prediction",
                    fallback_code=ErrorCode.INFER_PREDICTION_FAILED,
                    metadata={"stage": "prediction", "model_name": getattr(self.predictor, "model_name", None)},
                )
                return self._error_response(event)

            return {
                "status": "success",
                "claim_id": claim.get("claim_id"),
                "prediction": prediction,
                "features": features,
                "errors": [],
            }

        except Exception as exc:
            event = self.error_tracker.record_exception(
                exc,
                component="inference",
                stage="score_claim",
                fallback_code=ErrorCode.INFER_UNEXPECTED,
                metadata={"stage": "score_claim"},
            )
            return self._error_response(event)

    def _error_response(self, event: Any) -> dict[str, Any]:
        return {
            "status": "error",
            "error": {
                "event_id": event.event_id,
                "error_code": event.error_code,
                "message": event.message,
                "severity": event.severity,
                "category": event.category,
                "retryable": event.retryable,
                "occurrence_count": event.occurrence_count,
                "is_repeated": event.is_repeated,
            },
            "prediction": None,
            "features": None,
        }
