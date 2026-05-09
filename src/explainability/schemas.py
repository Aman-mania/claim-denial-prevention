"""Schemas and stable column names for Week 5 explainability outputs.

These constants are part of the Week 5 contract. Week 6 RAG should consume
these columns instead of reading model-specific internals directly.
"""

from __future__ import annotations

EXPLANATION_VERSION = "week5_xai_v2"

# Reason-level output: one claim can have multiple rows.
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
    "evidence_type",          # critical_rule | shap | fallback
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

# Claim-level output: one row per claim, optimized for dashboard + Week 6 handoff.
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

# Stable filenames/tables. Keeping these centralized helps when replacing local
# Parquet with Delta/Unity Catalog tables later.
GOLD_FEATURE_TABLE = "gold_claim_features"
EXPLANATION_TABLE = "gold_claim_explanations"
EXPLANATION_SUMMARY_TABLE = "gold_claim_explanation_summary"
EXPLANATION_REPORT_FILE = "explanation_report.json"
