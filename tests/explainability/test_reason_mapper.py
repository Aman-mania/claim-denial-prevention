from src.explainability.reason_mapper import ReasonMapper
from src.explainability.reason_catalog import get_reason_for_feature, policy_tags_for_reasons


def test_reason_catalog_maps_core_features():
    assert get_reason_for_feature("diagnosis_code_missing").reason_code == "MISSING_DIAGNOSIS"
    assert get_reason_for_feature("billed_deviation_imputed_capped").reason_code == "HIGH_BILLING_AMOUNT"
    assert "diagnosis" in policy_tags_for_reasons(["MISSING_DIAGNOSIS"])


def test_reason_mapper_deduplicates_multiple_features_for_same_reason():
    mapper = ReasonMapper(max_reasons=3)
    shap_explanation = {
        "top_reasons": [
            {
                "feature": "billed_deviation_imputed_capped",
                "label": "Billing Deviation",
                "shap_value": 2.0,
                "direction": "increases_risk",
            },
            {
                "feature": "is_high_cost",
                "label": "High Cost",
                "shap_value": 1.5,
                "direction": "increases_risk",
            },
            {
                "feature": "diagnosis_code_missing",
                "label": "Missing Diagnosis",
                "shap_value": 1.0,
                "direction": "increases_risk",
            },
        ]
    }
    claim_features = {
        "billed_deviation_imputed_capped": 150.0,
        "is_high_cost": 1,
        "diagnosis_code_missing": True,
    }
    prediction = {"risk_level": "HIGH"}

    rows = mapper.map(
        shap_explanation=shap_explanation,
        claim_features=claim_features,
        prediction=prediction,
    )

    codes = [r["reason_code"] for r in rows]
    assert codes == ["HIGH_BILLING_AMOUNT", "MISSING_DIAGNOSIS"]
    assert rows[0]["reason_rank"] == 1
    assert "150.0%" in rows[0]["reason_text"]
    assert rows[0]["policy_query"]


def test_reason_mapper_ignores_negative_reasons_for_risky_claims_when_possible():
    mapper = ReasonMapper(max_reasons=2)
    shap_explanation = {
        "top_reasons": [
            {
                "feature": "provider_claim_count",
                "shap_value": -2.0,
                "direction": "decreases_risk",
            },
            {
                "feature": "billed_amount_missing",
                "shap_value": 1.0,
                "direction": "increases_risk",
            },
        ]
    }
    rows = mapper.map(
        shap_explanation=shap_explanation,
        claim_features={"billed_amount_missing": True, "provider_claim_count": 2},
        prediction={"risk_level": "HIGH"},
    )
    assert len(rows) == 1
    assert rows[0]["reason_code"] == "MISSING_AMOUNT"
