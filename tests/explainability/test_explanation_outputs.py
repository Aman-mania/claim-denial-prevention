import pandas as pd

from src.explainability.explanation_generator import ExplanationGenerationPipeline
from src.explainability.schemas import EXPLANATION_COLUMNS, SUMMARY_COLUMNS


def test_build_summary_from_long_rows(tmp_path):
    pipeline = ExplanationGenerationPipeline(
        gold_dir=tmp_path,
        models_dir=tmp_path,
        output_dir=tmp_path,
    )
    long_df = pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "risk_score": 0.9,
                "risk_level": "HIGH",
                "predicted_denial": 1,
                "classification_threshold": 0.6,
                "review_threshold": 0.35,
                "reason_rank": 1,
                "reason_code": "MISSING_DIAGNOSIS",
                "reason_text": "Missing diagnosis.",
                "fix_suggestion": "Add diagnosis.",
                "policy_query": "diagnosis required",
                "policy_tags": '["diagnosis"]',
                "model_used": "xgboost",
                "created_at": "now",
            },
            {
                "claim_id": "C1",
                "risk_score": 0.9,
                "risk_level": "HIGH",
                "predicted_denial": 1,
                "classification_threshold": 0.6,
                "review_threshold": 0.35,
                "reason_rank": 2,
                "reason_code": "HIGH_BILLING_AMOUNT",
                "reason_text": "High billing.",
                "fix_suggestion": "Verify amount.",
                "policy_query": "billing documentation",
                "policy_tags": '["billing"]',
                "model_used": "xgboost",
                "created_at": "now",
            },
        ]
    )

    summary = pipeline._build_summary(long_df)

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 1
    assert summary.loc[0, "reason_1"] == "Missing diagnosis."
    assert summary.loc[0, "reason_2"] == "High billing."
    assert "MISSING_DIAGNOSIS" in summary.loc[0, "reason_codes"]


def test_explanation_schema_columns_are_stable():
    assert "claim_id" in EXPLANATION_COLUMNS
    assert "reason_code" in EXPLANATION_COLUMNS
    assert "policy_query" in EXPLANATION_COLUMNS
    assert "shap_output_unit" in EXPLANATION_COLUMNS
