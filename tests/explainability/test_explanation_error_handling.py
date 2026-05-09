import pytest

from src.explainability.reason_mapper import ReasonMapper
from src.explainability.service import ExplanationService
from src.observability import ErrorCode

pytestmark = [pytest.mark.unit, pytest.mark.week5]


class BrokenPredictor:
    def predict(self, features):
        raise RuntimeError("prediction broke")


class DummyExplainer:
    def explain(self, features, top_n=10):
        return {"top_reasons": []}


def test_explanation_service_returns_error_envelope_on_failure(tmp_path):
    from src.observability import ErrorTracker

    tracker = ErrorTracker(log_dir=tmp_path)
    service = ExplanationService(
        predictor=BrokenPredictor(),
        explainer=DummyExplainer(),
        reason_mapper=ReasonMapper(),
        error_tracker=tracker,
    )

    result = service.explain_feature_row({"claim_id": "CERR"})

    assert result["status"] == "error"
    assert result["error"]["error_code"] == ErrorCode.XAI_EXPLANATION_GENERATION_FAILED.value
    assert (tmp_path / "error_events.jsonl").exists()
