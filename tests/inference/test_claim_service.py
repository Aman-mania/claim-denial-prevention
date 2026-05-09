from pathlib import Path

from src.inference.claim_service import ClaimDenialService
from src.observability import ErrorCode, ErrorTracker


class DummyBuilder:
    def build_features(self, claim):
        return {
            "claim_id": claim["claim_id"],
            "diagnosis_code_missing": False,
            "procedure_code_missing": False,
            "billed_amount_missing": False,
        }


class DummyPredictor:
    model_name = "dummy"

    def predict(self, features):
        return {
            "claim_id": features["claim_id"],
            "risk_score": 0.25,
            "risk_level": "LOW",
            "predicted_denial": 0,
            "classification_threshold": 0.65,
            "review_threshold": 0.40,
            "model_used": "dummy",
        }


def test_claim_service_scores_valid_raw_claim(tmp_path: Path):
    service = ClaimDenialService(
        feature_builder=DummyBuilder(),
        predictor=DummyPredictor(),
        error_tracker=ErrorTracker(log_dir=tmp_path),
    )

    result = service.score_claim({
        "claim_id": "C001",
        "patient_id": "P001",
        "provider_id": "PR001",
        "diagnosis_code": "D10",
        "procedure_code": "PROC1",
        "billed_amount": 1000,
    })

    assert result["status"] == "success"
    assert result["prediction"]["risk_level"] == "LOW"
    assert result["features"]["claim_id"] == "C001"


def test_claim_service_rejects_inconsistent_amount_state(tmp_path: Path):
    service = ClaimDenialService(
        feature_builder=DummyBuilder(),
        predictor=DummyPredictor(),
        error_tracker=ErrorTracker(log_dir=tmp_path),
    )

    result = service.score_claim({
        "claim_id": "C001",
        "patient_id": "P001",
        "provider_id": "PR001",
        "billed_amount_missing": True,
        "billed_amount": 1000,
    })

    assert result["status"] == "error"
    assert result["error"]["error_code"] == ErrorCode.INFER_INCONSISTENT_CLAIM_STATE.value
