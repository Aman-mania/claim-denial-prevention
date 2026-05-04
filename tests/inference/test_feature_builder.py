from src.inference.feature_builder import CustomClaimFeatureBuilder


def test_custom_claim_feature_builder_handles_missing_amount():
    artifacts = {
        "amount_imputation": {
            "global_median": 10000.0,
            "procedure_medians": {"PROC1": 5000.0},
        },
        "cost": {
            "expected_cost_p75": 12000.0,
            "regional_lookup": {"PROC1|Delhi": {"expected_cost": 4000.0}},
            "procedure_lookup": {"PROC1": {"expected_cost": 4500.0}},
            "match_encoding": {"missing": 0, "procedure_avg": 1, "regional": 2},
        },
        "provider_history": {
            "PR1": {
                "specialty": "Cardiology",
                "location": "Delhi",
                "provider_claim_count": 7,
                "provider_violation_rate": 0.2,
            }
        },
        "patient_claim_counts": {"P1": 3},
        "diagnosis_lookup": {"D10": {"category": "Heart", "severity": "High", "severity_rank": 3}},
        "specialty_map": {"Cardiology": 2},
    }
    builder = CustomClaimFeatureBuilder(artifacts)
    features = builder.build({
        "claim_id": "CNEW",
        "patient_id": "P1",
        "provider_id": "PR1",
        "diagnosis_code": "D10",
        "procedure_code": "PROC1",
        "billed_amount": None,
    })

    assert features["billed_amount_missing"] is True
    assert features["billed_amount_imputed"] == 5000.0
    assert features["amount_imputation_strategy"] == "procedure_median"
    assert features["cost_match_level"] == "regional"
    assert features["cost_match_encoded"] == 2
