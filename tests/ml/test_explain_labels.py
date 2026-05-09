from src.ml.explain import FEATURE_LABELS, FIX_SUGGESTIONS


def test_explainability_labels_cover_current_gold_features():
    expected = {
        "billed_deviation_imputed_capped",
        "billed_amount_imputed",
        "log_billed_amount_imputed",
        "cost_match_encoded",
        "severity_rank",
    }

    assert expected.issubset(FEATURE_LABELS)
    assert expected.issubset(FIX_SUGGESTIONS)
