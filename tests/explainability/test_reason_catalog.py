import pytest

from src.explainability.reason_catalog import (
    CRITICAL_FEATURES,
    REASON_CATALOG,
    get_reason_for_feature,
    policy_tags_for_reasons,
)

pytestmark = [pytest.mark.unit, pytest.mark.week5]


def test_reason_catalog_maps_core_features():
    assert get_reason_for_feature("diagnosis_code_missing").reason_code == "MISSING_DIAGNOSIS"
    assert get_reason_for_feature("billed_deviation_imputed_capped").reason_code == "HIGH_BILLING_AMOUNT"
    assert "diagnosis" in policy_tags_for_reasons(["MISSING_DIAGNOSIS"])


def test_reason_catalog_has_policy_query_and_fix_for_each_reason():
    for reason in REASON_CATALOG.values():
        assert reason.reason_code
        assert reason.title
        assert reason.fix_suggestion
        assert reason.policy_query_template
        assert reason.policy_tags
        assert reason.source_features


def test_critical_features_are_explicitly_registered():
    assert "diagnosis_code_missing" in CRITICAL_FEATURES
    assert "procedure_code_missing" in CRITICAL_FEATURES
    assert "billed_amount_missing" in CRITICAL_FEATURES
    assert "proc_no_diag" in CRITICAL_FEATURES
