import pandas as pd

from src.io.table_contracts import enforce_table_contract


def test_gold_feature_contract_downcasts_binary_and_encoded_columns():
    df = pd.DataFrame({
        "claim_id": ["C1", "C2"],
        "diagnosis_code_missing": [0, 1],
        "procedure_code_missing": [False, True],
        "billed_amount_missing": [0, 0],
        "proc_no_diag": [1, 0],
        "diag_no_proc": [0, 1],
        "is_high_cost": [0, 1],
        "cost_match_encoded": [0, 2],
        "severity_rank": [1, 3],
        "specialty_encoded": [1, 4],
        "provider_claim_count": [48, 63],
        "patient_claim_count": [2, 8],
        "billed_amount_imputed": [10000.0, 22000.0],
        "billed_deviation_imputed_capped": [12.5, 150.0],
        "provider_violation_rate": [0.1, 1.28],
    })

    out = enforce_table_contract(df, "gold_claim_features")

    assert str(out["diagnosis_code_missing"].dtype) == "bool"
    assert str(out["is_high_cost"].dtype) == "bool"
    assert str(out["cost_match_encoded"].dtype) == "int8"
    assert str(out["severity_rank"].dtype) == "int8"
    assert str(out["provider_claim_count"].dtype) == "int32"
    assert str(out["billed_amount_imputed"].dtype) == "float32"


def test_explanation_contract_serializes_mixed_feature_value_to_string():
    df = pd.DataFrame({
        "claim_id": ["C1", "C2"],
        "risk_score": [0.82, 0.12],
        "risk_level": ["HIGH", "LOW"],
        "predicted_denial": [1, 0],
        "classification_threshold": [0.62, 0.62],
        "review_threshold": [0.35, 0.35],
        "reason_rank": [1, 1],
        "reason_code": ["MISSING_DIAGNOSIS", "HIGH_BILLING_AMOUNT"],
        "reason_title": ["Missing diagnosis", "High billing"],
        "reason_text": ["Diagnosis missing", "Amount high"],
        "business_category": ["claim_completeness", "billing"],
        "evidence_type": ["critical_rule", "shap"],
        "feature_name": ["diagnosis_code_missing", "billed_deviation_imputed_capped"],
        "feature_label": ["Missing Diagnosis Code", "Billing Deviation"],
        "feature_value": [True, 112.75],
        "shap_value": [3.2, 1.4],
        "shap_direction": ["increases_risk", "increases_risk"],
        "shap_output_unit": ["raw_log_odds", "raw_log_odds"],
        "fix_suggestion": ["Add diagnosis", "Review amount"],
        "policy_query": ["diagnosis required", "cost justification"],
        "policy_tags": ['["diagnosis"]', '["billing"]'],
        "model_used": ["xgboost", "xgboost"],
        "explanation_version": ["test", "test"],
        "created_at": ["now", "now"],
    })

    out = enforce_table_contract(df, "gold_claim_explanations")

    assert str(out["feature_value"].dtype) == "string"
    assert out["feature_value"].tolist() == ["True", "112.75"]
    assert str(out["shap_value"].dtype) == "float32"
    assert str(out["predicted_denial"].dtype) == "int8"


def test_unknown_table_is_unchanged():
    df = pd.DataFrame({"x": [1, 2]})
    out = enforce_table_contract(df, "some_future_table")
    assert out.equals(df)
