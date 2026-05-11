import pandas as pd

from tools.audit_tables import audit_table


def test_audit_table_detects_contract_and_suggestions(tmp_path):
    path = tmp_path / "gold_claim_features.parquet"
    df = pd.DataFrame({
        "claim_id": ["C1", "C2"],
        "is_high_cost": [0, 1],
        "cost_match_encoded": [0, 2],
        "billed_amount_imputed": [100.0, 200.0],
    })
    df.to_parquet(path, index=False)

    report = audit_table(path)

    assert report["table"] == "gold_claim_features"
    assert report["has_contract"] is True
    issue_cols = {c["column"]: c for c in report["columns_detail"]}
    assert issue_cols["is_high_cost"]["contracted_dtype"] == "bool"
    assert issue_cols["cost_match_encoded"]["contracted_dtype"] == "int8"
