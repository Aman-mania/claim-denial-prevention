"""Central table dtype contracts for local Parquet and future Delta tables.

Every new persisted table should get a contract here before being written. This
prevents PyArrow/Delta schema drift, keeps binary fields boolean/int8 instead of
int64, and reduces storage cost before migration to Databricks.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

# Known table dtype contracts. Use pandas extension dtypes where nulls are valid.
TABLE_CONTRACTS: dict[str, dict[str, str]] = {
    "claims_silver": {
        "claim_id": "string", "patient_id": "string", "provider_id": "string",
        "diagnosis_code": "string", "procedure_code": "string", "billed_amount": "float64",
        "denial_flag": "Int8",
        "diagnosis_code_missing": "bool", "procedure_code_missing": "bool", "billed_amount_missing": "bool",
        "proc_no_diag": "bool", "diag_no_proc": "bool",
    },
    "providers_silver": {
        "provider_id": "string", "doctor_name": "string", "specialty": "string", "location": "string",
        "location_missing": "bool",
    },
    "diagnosis_silver": {
        "diagnosis_code": "string", "category": "string", "severity": "string",
    },
    "cost_silver": {
        "procedure_code": "string", "average_cost": "float64", "expected_cost": "float64",
        "region": "string", "cost_ratio": "float32",
    },
    "gold_claim_base": {
        "claim_id": "string", "patient_id": "string", "provider_id": "string",
        "diagnosis_code": "string", "procedure_code": "string", "billed_amount": "float64",
        "denial_flag": "Int8", "denial_risk_score": "float32",
        "diagnosis_code_missing": "bool", "procedure_code_missing": "bool", "billed_amount_missing": "bool",
        "proc_no_diag": "bool", "diag_no_proc": "bool",
        "specialty": "string", "location": "string", "category": "string", "severity": "string",
        "expected_cost": "float64", "average_cost": "float64", "cost_ratio": "float32",
        "billed_deviation_pct": "float32", "cost_match_level": "string", "label_source": "string",
    },
    "gold_claim_features": {
        "claim_id": "string", "patient_id": "string", "provider_id": "string",
        "diagnosis_code": "string", "procedure_code": "string", "billed_amount": "float64",
        "denial_flag": "Int8", "denial_risk_score": "float32",
        "diagnosis_code_missing": "bool", "procedure_code_missing": "bool", "billed_amount_missing": "bool",
        "proc_no_diag": "bool", "diag_no_proc": "bool",
        "provider_claim_count": "int32", "provider_avg_billed": "float32", "provider_violation_rate": "float32",
        "patient_claim_count": "int32", "severity_encoded": "int8", "severity_rank": "int8",
        "specialty_encoded": "int8", "billed_amount_imputed": "float32",
        "log_billed_amount_imputed": "float32", "log_billed_amount": "float32",
        "billed_deviation_capped": "float32", "billed_deviation_imputed_pct": "float32",
        "billed_deviation_imputed_capped": "float32", "is_high_cost": "bool",
        "cost_match_encoded": "int8", "amount_imputation_strategy": "string",
        "cost_match_level": "string", "label_source": "string",
    },
    "gold_claim_explanations": {
        "claim_id": "string", "risk_score": "float32", "risk_level": "string",
        "predicted_denial": "Int8", "classification_threshold": "float32", "review_threshold": "float32",
        "reason_rank": "Int8", "reason_code": "string", "reason_title": "string", "reason_text": "string",
        "business_category": "string", "evidence_type": "string", "feature_name": "string",
        "feature_label": "string", "feature_value": "string", "shap_value": "float32",
        "shap_direction": "string", "shap_output_unit": "string", "fix_suggestion": "string",
        "policy_query": "string", "policy_tags": "string", "model_used": "string",
        "explanation_version": "string", "created_at": "string",
    },
    "gold_claim_explanation_summary": {
        "claim_id": "string", "risk_score": "float32", "risk_level": "string",
        "predicted_denial": "Int8", "classification_threshold": "float32", "review_threshold": "float32",
        "reason_1": "string", "reason_2": "string", "reason_3": "string",
        "reason_codes": "string", "reason_texts_json": "string", "fix_suggestions_json": "string",
        "policy_queries_json": "string", "policy_tags_json": "string", "model_used": "string",
        "explanation_version": "string", "created_at": "string",
    },
    "policy_chunks": {
        "chunk_id": "string", "document_id": "string", "source_name": "string", "source_type": "string",
        "source_path": "string", "section_title": "string", "page_number": "Int32", "chunk_index": "int32",
        "chunk_text": "string", "policy_tags_json": "string", "token_estimate": "int32",
        "embedding_model": "string", "rag_version": "string", "created_at": "string",
    },
    "gold_claim_policy_matches": {
        "claim_id": "string", "risk_score": "float32", "risk_level": "string", "predicted_denial": "Int8",
        "reason_rank": "Int8", "reason_code": "string", "reason_title": "string", "reason_text": "string",
        "fix_suggestion": "string", "policy_rank": "Int8", "policy_chunk_id": "string", "policy_text": "string",
        "policy_summary": "string", "source_name": "string", "source_type": "string", "source_path": "string",
        "section_title": "string", "page_number": "Int32", "similarity_score": "float32", "raw_similarity_score": "float32",
        "tag_overlap_count": "Int16", "retrieval_query": "string", "query_policy_tags_json": "string",
        "retrieved_policy_tags_json": "string", "rag_version": "string", "created_at": "string",
    },
    "gold_claim_final_explanations": {
        "claim_id": "string", "risk_score": "float32", "risk_level": "string", "predicted_denial": "Int8",
        "final_explanation_text": "string", "reasons_json": "string", "policies_json": "string",
        "recommended_actions_json": "string", "rag_version": "string", "created_at": "string",
    },
}


def get_table_contract(table_name: str) -> Mapping[str, str]:
    """Return a dtype contract for table_name, or an empty mapping."""
    return TABLE_CONTRACTS.get(table_name, {})


def enforce_table_contract(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Return a copy of df with known columns coerced to the table contract.

    Unknown columns are preserved. Missing contract columns are not created here;
    the owning pipeline should create expected schema columns when needed.
    """
    contract = get_table_contract(table_name)
    if not contract or df.empty:
        return df

    out = df.copy()
    for column, dtype in contract.items():
        if column not in out.columns:
            continue
        if dtype in {"float32", "float64"}:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(dtype)
        elif dtype in {"int8", "int16", "int32", "int64"}:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(dtype)
        elif dtype in {"Int8", "Int16", "Int32", "Int64"}:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(dtype)
        elif dtype == "bool":
            out[column] = out[column].fillna(False).astype(bool)
        elif dtype == "string":
            out[column] = out[column].astype("string")
        else:
            out[column] = out[column].astype(dtype)
    return out


def summarize_contract_issues(df: pd.DataFrame, table_name: str) -> list[dict[str, str]]:
    """Return human-readable dtype mismatches against a table contract."""
    issues: list[dict[str, str]] = []
    contract = get_table_contract(table_name)
    if not contract:
        return issues
    for column, expected in contract.items():
        if column not in df.columns:
            continue
        actual = str(df[column].dtype)
        expected_norm = expected.lower()
        actual_norm = actual.lower()
        if expected_norm == "string" and actual_norm in {"string", "object"}:
            continue
        if expected_norm != actual_norm:
            issues.append({"column": column, "expected": expected, "actual": actual})
    return issues
