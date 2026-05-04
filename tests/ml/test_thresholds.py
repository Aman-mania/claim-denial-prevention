import numpy as np

from src.ml.train import _risk_band_policy, _tune_threshold


def test_threshold_tuning_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    selected, rows = _tune_threshold(y_true, y_prob, min_recall=0.9)

    assert 0.05 <= selected["threshold"] <= 0.95
    assert selected["recall"] >= 0.9
    assert len(rows) > 10


def test_risk_band_policy_orders_thresholds():
    policy = _risk_band_policy(0.55)
    assert policy["medium_lower_inclusive"] < policy["high_lower_inclusive"]
    assert policy["classification_threshold"] == policy["high_lower_inclusive"]
