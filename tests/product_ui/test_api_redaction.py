import pytest

pytest.importorskip("fastapi")

from api.routes.claims import _redact_for_role


def test_analyst_response_hides_features_and_debug_reason_fields():
    result = {
        "status": "success",
        "features": {"secret_feature": 1},
        "prediction": {"risk_level": "HIGH", "risk_score": 0.9, "features_received": 15},
        "reasons": [{
            "reason_code": "MISSING_DIAGNOSIS",
            "reason_title": "Missing diagnosis",
            "reason_text": "Missing diagnosis.",
            "reason_definition": {"debug": True},
            "shap_value": 2.2,
        }],
        "policy_evidence": [{
            "reason_code": "MISSING_DIAGNOSIS",
            "source_name": "policy.md",
            "source_path": "/local/path/policy.md",
            "raw_similarity_score": 0.8,
            "similarity_score": 0.9,
        }],
    }

    redacted = _redact_for_role(result, "analyst")

    assert "features" not in redacted
    assert "features_received" not in redacted["prediction"]
    assert "reason_definition" not in redacted["reasons"][0]
    assert "shap_value" not in redacted["reasons"][0]
    assert "source_path" not in redacted["policy_evidence"][0]
    assert "raw_similarity_score" not in redacted["policy_evidence"][0]
    assert redacted["response_scope"] == "business_analyst"


def test_developer_response_keeps_full_payload():
    result = {"features": {"x": 1}, "reasons": [{"shap_value": 1.0}]}
    assert _redact_for_role(result, "developer") is result
