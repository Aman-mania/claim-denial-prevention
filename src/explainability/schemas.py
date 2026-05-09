"""Schemas and stable column names for Week 5 explainability outputs."""

from __future__ import annotations

EXPLANATION_VERSION = "week5_xai_v1"

EXPLANATION_COLUMNS: list[str] = [
    "claim_id",
    "risk_score",
    "risk_level",
    "predicted_denial",
    "classification_threshold",
    "review_threshold",
    "reason_rank",
    "reason_code",
    "reason_title",
    "reason_text",
    "business_category",
    "feature_name",
    "feature_label",
    "feature_value",
    "shap_value",
    "shap_direction",
    "shap_output_unit",
    "fix_suggestion",
    "policy_query",
    "policy_tags",
    "model_used",
    "explanation_version",
    "created_at",
]

SUMMARY_COLUMNS: list[str] = [
    "claim_id",
    "risk_score",
    "risk_level",
    "predicted_denial",
    "classification_threshold",
    "review_threshold",
    "reason_1",
    "reason_2",
    "reason_3",
    "reason_codes",
    "reason_texts_json",
    "fix_suggestions_json",
    "policy_queries_json",
    "policy_tags_json",
    "model_used",
    "explanation_version",
    "created_at",
]
