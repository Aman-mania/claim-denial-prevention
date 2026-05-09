import pandas as pd
import pytest

from src.explainability.explanation_generator import ExplanationGenerationPipeline
from src.explainability.schemas import EXPLANATION_COLUMNS, SUMMARY_COLUMNS

pytestmark = [pytest.mark.unit, pytest.mark.week5]


def test_build_summary_from_long_rows(tmp_path):
    pipeline = ExplanationGenerationPipeline(gold_dir=tmp_path, models_dir=tmp_path, output_dir=tmp_path)
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
    assert "evidence_type" in EXPLANATION_COLUMNS


def test_explanation_long_output_serializes_mixed_feature_values_for_parquet(tmp_path):
    """A single evidence column may contain bool/float/string source values.

    Parquet requires one physical type per column, so feature_value is stored as
    text while risk/SHAP columns stay numeric.
    """
    pipeline = ExplanationGenerationPipeline(gold_dir=tmp_path, models_dir=tmp_path, output_dir=tmp_path)
    long_df = pipeline._normalize_long_df([
        {
            "claim_id": "C1",
            "risk_score": 0.9,
            "risk_level": "HIGH",
            "predicted_denial": 1,
            "classification_threshold": 0.6,
            "review_threshold": 0.35,
            "reason_rank": 1,
            "reason_code": "MISSING_DIAGNOSIS",
            "reason_title": "Missing diagnosis",
            "reason_text": "Missing diagnosis.",
            "business_category": "claim_completeness",
            "evidence_type": "critical_rule",
            "feature_name": "diagnosis_code_missing",
            "feature_label": "Missing Diagnosis Code",
            "feature_value": True,
            "shap_value": 4.2,
            "shap_direction": "increases_risk",
            "shap_output_unit": "raw_log_odds_contribution",
            "fix_suggestion": "Add diagnosis.",
            "policy_query": "diagnosis required",
            "policy_tags": ["diagnosis"],
            "model_used": "xgboost",
            "explanation_version": "week5_xai_v2",
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
            "reason_title": "High billing",
            "reason_text": "High billing.",
            "business_category": "billing",
            "evidence_type": "shap",
            "feature_name": "billed_deviation_imputed_capped",
            "feature_label": "Billing Deviation",
            "feature_value": 112.75,
            "shap_value": 1.7,
            "shap_direction": "increases_risk",
            "shap_output_unit": "raw_log_odds_contribution",
            "fix_suggestion": "Verify amount.",
            "policy_query": "billing documentation",
            "policy_tags": ["billing"],
            "model_used": "xgboost",
            "explanation_version": "week5_xai_v2",
            "created_at": "now",
        },
    ])

    assert str(long_df["feature_value"].dtype) == "string"
    assert long_df.loc[0, "feature_value"] == "True"
    assert long_df.loc[1, "feature_value"] == "112.75"

    summary_df = pipeline._build_summary(long_df)
    long_path, summary_path = pipeline._write_outputs(long_df, summary_df)

    assert long_path.exists()
    assert summary_path.exists()
    reloaded = pd.read_parquet(long_path)
    assert reloaded.loc[1, "feature_value"] == "112.75"
