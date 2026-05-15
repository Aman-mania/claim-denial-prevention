from src.agent.openai_output import deterministic_presentation
from src.agent.recommendation_catalog import RecommendationCatalog


def test_recommendation_catalog_uses_reason_codes():
    catalog = RecommendationCatalog()
    validation = {"blocking_errors": [], "warnings": []}
    prediction = {"risk_level": "HIGH", "risk_score": 0.82}
    reasons = [{
        "reason_code": "MISSING_DIAGNOSIS",
        "reason_title": "Diagnosis is missing",
        "reason_text": "The claim is missing diagnosis support.",
        "fix_suggestion": "Add ICD code.",
    }]
    evidence = [{"reason_code": "MISSING_DIAGNOSIS", "source_name": "policy.md"}]

    recs = catalog.generate(validation=validation, prediction=prediction, reasons=reasons, policy_evidence=evidence)
    assert recs
    assert recs[0]["priority"] == "HIGH"
    assert "diagnosis" in recs[0]["action"].lower()
    decision = catalog.decision(validation=validation, prediction=prediction, recommendations=recs)
    assert decision["status"] == "REVIEW_REQUIRED"


def test_catalog_blocks_when_validation_has_errors():
    catalog = RecommendationCatalog()
    validation = {
        "blocking_errors": [{"code": "CLAIM_ERR_REQUIRED_FIELD_MISSING", "field": "provider_id", "message": "provider_id is required"}],
        "warnings": [],
    }
    recs = catalog.generate(validation=validation, prediction=None, reasons=[], policy_evidence=[])
    assert recs[0]["priority"] == "HIGH"
    decision = catalog.decision(validation=validation, prediction=None, recommendations=recs)
    assert decision["status"] == "BLOCKED"


def test_deterministic_presentation_contains_action_plan():
    payload = {
        "claim_id": "C00058",
        "prediction": {"risk_level": "HIGH", "risk_score": 0.82},
        "validation": {"blocking_errors": []},
        "decision": {"summary": "Review required."},
        "reasons": [{"reason_title": "High billing", "reason_text": "Amount is above benchmark."}],
        "policy_evidence": [{"source_name": "policy.md", "policy_summary": "High-cost claims require documentation."}],
        "recommendations": [{"action": "Review billed amount."}],
    }
    presentation = deterministic_presentation(payload)
    assert presentation["summary"]
    assert presentation["action_plan"] == ["Review billed amount."]
    assert presentation["source"] == "deterministic"
