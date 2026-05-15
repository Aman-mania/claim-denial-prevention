from src.agent.remediation_agent import RemediationAgent
from src.agent.recommendation_catalog import RecommendationCatalog


class _ValidationResult:
    is_valid = True
    normalized_claim = {
        "claim_id": "LOW001",
        "patient_id": "P001",
        "provider_id": "PR100",
        "diagnosis_code": "D001",
        "procedure_code": "PROC1",
        "billed_amount": 3000,
    }

    def to_dict(self):
        return {
            "is_valid": True,
            "normalized_claim": dict(self.normalized_claim),
            "blocking_errors": [],
            "warnings": [],
            "infos": [],
        }


class _Validator:
    def validate(self, claim):
        return _ValidationResult()


class _ClaimService:
    def score_claim(self, claim):
        return {
            "status": "success",
            "prediction": {
                "claim_id": claim["claim_id"],
                "risk_score": 0.011,
                "risk_level": "LOW",
                "predicted_denial": 0,
            },
            "features": {
                "claim_id": claim["claim_id"],
                "is_high_cost": 0,
                "cost_match_encoded": 1,
                "specialty_encoded": 1,
            },
        }


class _Explainer:
    def explain(self, features, top_n=10):
        return {"top_reasons": [{"feature": "is_high_cost", "shap_value": 0.0002, "direction": "increases_risk"}]}


class _ReasonMapper:
    def map(self, *, shap_explanation, claim_features, prediction):
        return [{
            "reason_rank": 1,
            "reason_code": "HIGH_BILLING_AMOUNT",
            "reason_title": "Billed amount above expected benchmark",
            "reason_text": "The billed amount is materially higher than expected.",
            "fix_suggestion": "Review amount.",
            "policy_query": "high billing amount",
            "policy_tags": ["billing"],
        }]


class _Retriever:
    called = False

    def retrieve(self, **kwargs):
        self.called = True
        return []


def test_low_risk_claim_suppresses_shap_risk_reasons_and_rag():
    retriever = _Retriever()
    agent = RemediationAgent(
        validator=_Validator(),
        claim_service=_ClaimService(),
        shap_explainer=_Explainer(),
        reason_mapper=_ReasonMapper(),
        retriever=retriever,
        recommendation_catalog=RecommendationCatalog(),
    )

    result = agent.analyze_claim({"claim_id": "LOW001"})

    assert result["status"] == "success"
    assert result["prediction"]["risk_level"] == "LOW"
    assert result["reasons"] == []
    assert result["policy_evidence"] == []
    assert retriever.called is False
    assert result["decision"]["status"] == "READY_TO_SUBMIT"
    assert "No blocking" in result["decision"]["summary"]
