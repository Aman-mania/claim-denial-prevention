import pytest

from src.explainability.reason_mapper import ReasonMapper
from src.explainability.service import ExplanationService

pytestmark = [pytest.mark.unit, pytest.mark.week5]


class DummyPredictor:
    model_name = "xgboost"

    def predict(self, features):
        return {
            "claim_id": features.get("claim_id"),
            "risk_score": 0.91,
            "risk_level": "HIGH",
            "predicted_denial": 1,
            "classification_threshold": 0.6,
            "review_threshold": 0.35,
            "model_used": "xgboost",
        }


class DummyExplainer:
    def explain(self, features, top_n=10):
        return {
            "top_reasons": [
                {"feature": "billed_deviation_imputed_capped", "shap_value": 2.0, "direction": "increases_risk"},
                {"feature": "provider_violation_rate", "shap_value": 1.0, "direction": "increases_risk"},
            ]
        }


class DummyFeatureBuilder:
    def build(self, raw_claim):
        return {
            "claim_id": raw_claim["claim_id"],
            "diagnosis_code_missing": True,
            "billed_deviation_imputed_capped": 120.0,
            "provider_violation_rate": 1.1,
        }


def test_explanation_service_explains_feature_row():
    service = ExplanationService(
        predictor=DummyPredictor(),
        explainer=DummyExplainer(),
        reason_mapper=ReasonMapper(max_reasons=3),
    )
    result = service.explain_feature_row({
        "claim_id": "C1",
        "diagnosis_code_missing": True,
        "billed_deviation_imputed_capped": 120.0,
        "provider_violation_rate": 1.1,
    })

    assert result["status"] == "success"
    assert result["prediction"]["risk_level"] == "HIGH"
    assert result["reasons"][0]["reason_code"] == "MISSING_DIAGNOSIS"


def test_explanation_service_explains_raw_claim_with_feature_builder():
    service = ExplanationService(
        predictor=DummyPredictor(),
        explainer=DummyExplainer(),
        reason_mapper=ReasonMapper(max_reasons=3),
        feature_builder=DummyFeatureBuilder(),
    )
    result = service.explain_raw_claim({"claim_id": "C2"})

    assert result["status"] == "success"
    assert result["features"]["diagnosis_code_missing"] is True
    assert any(r["reason_code"] == "MISSING_DIAGNOSIS" for r in result["reasons"])
