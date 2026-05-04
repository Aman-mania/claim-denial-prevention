"""
Tests — Replacement Dataset Support
====================================
Focused tests for Week 4 updates introduced for the new supervised dataset:
  - real denial_flag is preserved
  - regional cost table does not duplicate claim rows
  - median-imputed amount features are added without overwriting billed_amount
"""

import pandas as pd

from src.gold.features import GoldFeaturePipeline
from src.constants import (
    SENTINEL_MISSING,
    COL_DIAG_MISSING,
    COL_PROC_MISSING,
    COL_AMOUNT_MISSING,
    COL_PROC_NO_DIAG,
    COL_DIAG_NO_PROC,
    COL_AMOUNT_IMPUTED,
    COL_AMOUNT_IMPUTATION_STRATEGY,
    COL_COST_MATCH_LEVEL,
    COL_LABEL_SOURCE,
)


def _claims_with_real_label():
    return pd.DataFrame({
        "claim_id": ["C001", "C002", "C003"],
        "patient_id": ["P001", "P002", "P003"],
        "provider_id": ["PR100", "PR101", "PR100"],
        "diagnosis_code": ["D10", "D20", SENTINEL_MISSING],
        "procedure_code": ["PROC1", "PROC1", "PROC2"],
        "billed_amount": [5000.0, None, 9000.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "denial_flag": [0, 1, 1],
        COL_DIAG_MISSING: [False, False, True],
        COL_PROC_MISSING: [False, False, False],
        COL_AMOUNT_MISSING: [False, True, False],
        COL_PROC_NO_DIAG: [False, False, True],
        COL_DIAG_NO_PROC: [False, False, False],
    })


def _providers():
    return pd.DataFrame({
        "provider_id": ["PR100", "PR101"],
        "specialty": ["Cardiology", "Neurology"],
        "location": ["Delhi", "Mumbai"],
    })


def _diagnosis():
    return pd.DataFrame({
        "diagnosis_code": ["D10", "D20"],
        "category": ["Heart", "Bone"],
        "severity": ["High", "Medium"],
    })


def _regional_cost():
    # Two rows for PROC1 intentionally mirrors replacement/cost.csv.
    # The Gold join must still output one row per claim.
    return pd.DataFrame({
        "procedure_code": ["PROC1", "PROC1", "PROC2"],
        "expected_cost": [5000.0, 5200.0, 9000.0],
        "average_cost": [4500.0, 4600.0, 8000.0],
        "region": ["Delhi", "Mumbai", "Delhi"],
        "cost_ratio": [0.9, 0.8846, 0.8889],
    })


def test_replacement_cost_join_preserves_one_row_per_claim(tmp_path):
    pipeline = GoldFeaturePipeline(tmp_path / "silver", tmp_path / "gold")
    base = pipeline._build_base(_claims_with_real_label(), _providers(), _diagnosis(), _regional_cost())

    assert len(base) == 3
    assert base["claim_id"].is_unique
    assert set(base[COL_COST_MATCH_LEVEL]).issubset({"regional", "procedure_avg", "missing"})


def test_real_denial_flag_is_preserved(tmp_path):
    pipeline = GoldFeaturePipeline(tmp_path / "silver", tmp_path / "gold")
    base = pipeline._build_base(_claims_with_real_label(), _providers(), _diagnosis(), _regional_cost())
    labelled = pipeline._create_denial_label(base)

    assert labelled["denial_flag"].tolist() == [0, 1, 1]
    assert labelled[COL_LABEL_SOURCE].eq("provided").all()


def test_amount_imputation_adds_feature_without_overwriting_raw_amount(tmp_path):
    pipeline = GoldFeaturePipeline(tmp_path / "silver", tmp_path / "gold")
    base = pipeline._build_base(_claims_with_real_label(), _providers(), _diagnosis(), _regional_cost())
    labelled = pipeline._create_denial_label(base)
    features = pipeline._build_features(labelled)

    missing_raw = features.loc[features["claim_id"] == "C002"]
    assert pd.isna(missing_raw["billed_amount"].iloc[0])
    assert pd.notna(missing_raw[COL_AMOUNT_IMPUTED].iloc[0])
    assert missing_raw[COL_AMOUNT_IMPUTATION_STRATEGY].iloc[0] in {"procedure_median", "global_median"}
