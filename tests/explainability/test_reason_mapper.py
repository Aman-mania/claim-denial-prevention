import pytest

from src.explainability.reason_mapper import ReasonMapper

pytestmark = [pytest.mark.unit, pytest.mark.week5]


def test_reason_mapper_deduplicates_multiple_features_for_same_reason():
    mapper = ReasonMapper(max_reasons=3)
    shap_explanation = {
        "top_reasons": [
            {"feature": "billed_deviation_imputed_capped", "shap_value": 2.0, "direction": "increases_risk"},
            {"feature": "is_high_cost", "shap_value": 1.5, "direction": "increases_risk"},
            {"feature": "diagnosis_code_missing", "shap_value": 1.0, "direction": "increases_risk"},
        ]
    }
    claim_features = {
        "billed_deviation_imputed_capped": 150.0,
        "is_high_cost": 1,
        "diagnosis_code_missing": False,
    }
    prediction = {"risk_level": "HIGH"}

    rows = mapper.map(shap_explanation=shap_explanation, claim_features=claim_features, prediction=prediction)

    codes = [r["reason_code"] for r in rows]
    assert codes == ["HIGH_BILLING_AMOUNT", "MISSING_DIAGNOSIS"]
    assert rows[0]["reason_rank"] == 1
    assert "150.0%" in rows[0]["reason_text"]
    assert rows[0]["policy_query"]


def test_critical_rule_reason_appears_even_when_not_top_shap():
    mapper = ReasonMapper(max_reasons=3)
    shap_explanation = {
        "top_reasons": [
            {"feature": "provider_violation_rate", "shap_value": 3.5, "direction": "increases_risk"},
            {"feature": "billed_deviation_imputed_capped", "shap_value": 2.0, "direction": "increases_risk"},
        ]
    }
    claim_features = {
        "diagnosis_code_missing": True,
        "provider_violation_rate": 1.3,
        "billed_deviation_imputed_capped": 100.0,
    }

    rows = mapper.map(shap_explanation=shap_explanation, claim_features=claim_features, prediction={"risk_level": "HIGH"})

    assert rows[0]["reason_code"] == "MISSING_DIAGNOSIS"
    assert rows[0]["evidence_type"] == "critical_rule"


def test_reason_mapper_tracks_unmapped_features():
    mapper = ReasonMapper(max_reasons=2)
    rows = mapper.map(
        shap_explanation={"top_reasons": [{"feature": "unknown_model_feature", "shap_value": 1.2}]},
        claim_features={},
        prediction={"risk_level": "HIGH"},
    )
    assert rows == []
    assert mapper.last_unmapped_features == ["unknown_model_feature"]
