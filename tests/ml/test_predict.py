"""
Tests — ML Predict and Explain
================================
Tests for ClaimPredictor and SHAPExplainer.
Uses a minimal trained model to avoid loading real pkl files in tests.
"""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ml.predict import ClaimPredictor, _risk_level
from src.ml.explain import SHAPExplainer, FIX_SUGGESTIONS, FEATURE_LABELS


# ── Minimal fixtures ──────────────────────────────────────────────────────────

SAMPLE_FEATURES = [
    "diagnosis_code_missing", "procedure_code_missing", "billed_amount_missing",
    "proc_no_diag", "diag_no_proc", "billed_deviation_capped",
    "log_billed_amount", "is_high_cost", "provider_claim_count",
    "provider_violation_rate", "patient_claim_count",
    "severity_encoded", "specialty_encoded",
]


def _make_mock_pipeline(prob: float = 0.75):
    """Create a mock sklearn pipeline that returns a fixed probability."""
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[1 - prob, prob]])
    pipeline.predict.return_value       = np.array([int(prob >= 0.5)])
    return pipeline


@pytest.fixture
def mock_predictor():
    pipeline = _make_mock_pipeline(prob=0.75)
    return ClaimPredictor(
        pipeline=pipeline,
        features=SAMPLE_FEATURES,
        model_name="xgboost",
    )


@pytest.fixture
def sample_claim():
    """A complete claim feature dict — all fields present."""
    return {
        "claim_id":                "C0001",
        "diagnosis_code_missing":  False,
        "procedure_code_missing":  False,
        "billed_amount_missing":   False,
        "proc_no_diag":            False,
        "diag_no_proc":            False,
        "billed_deviation_capped": 325.0,
        "log_billed_amount":       9.8,
        "is_high_cost":            1,
        "provider_claim_count":    52,
        "provider_violation_rate": 1.2,
        "patient_claim_count":     2,
        "severity_encoded":        2,
        "specialty_encoded":       2,
    }


@pytest.fixture
def incomplete_claim():
    """A claim with several missing features."""
    return {
        "claim_id":               "C9999",
        "proc_no_diag":           True,
        "diagnosis_code_missing": True,
        # All other features absent — should be handled by imputer
    }


# ── _risk_level threshold tests ───────────────────────────────────────────────

class TestRiskLevel:
    def test_high_threshold(self):
        assert _risk_level(0.65) == "HIGH"
        assert _risk_level(0.99) == "HIGH"
        assert _risk_level(1.0)  == "HIGH"

    def test_medium_threshold(self):
        assert _risk_level(0.40) == "MEDIUM"
        assert _risk_level(0.55) == "MEDIUM"
        assert _risk_level(0.64) == "MEDIUM"

    def test_low_threshold(self):
        assert _risk_level(0.0)  == "LOW"
        assert _risk_level(0.20) == "LOW"
        assert _risk_level(0.39) == "LOW"

    def test_boundary_exactly_at_threshold(self):
        assert _risk_level(0.65) == "HIGH"   # >= 0.65
        assert _risk_level(0.40) == "MEDIUM" # >= 0.40


# ── ClaimPredictor tests ──────────────────────────────────────────────────────

class TestClaimPredictor:
    def test_predict_returns_required_keys(self, mock_predictor, sample_claim):
        result = mock_predictor.predict(sample_claim)
        for key in ["claim_id", "risk_score", "risk_level", "model_used",
                    "features_received", "features_expected"]:
            assert key in result

    def test_predict_risk_score_is_float(self, mock_predictor, sample_claim):
        result = mock_predictor.predict(sample_claim)
        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_predict_risk_level_valid(self, mock_predictor, sample_claim):
        result = mock_predictor.predict(sample_claim)
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_predict_claim_id_preserved(self, mock_predictor, sample_claim):
        result = mock_predictor.predict(sample_claim)
        assert result["claim_id"] == "C0001"

    def test_predict_model_name_in_result(self, mock_predictor, sample_claim):
        result = mock_predictor.predict(sample_claim)
        assert result["model_used"] == "xgboost"

    def test_predict_high_prob_gives_high_risk(self, sample_claim):
        predictor = ClaimPredictor(
            pipeline=_make_mock_pipeline(prob=0.90),
            features=SAMPLE_FEATURES,
            model_name="xgboost",
        )
        result = predictor.predict(sample_claim)
        assert result["risk_level"] == "HIGH"
        assert result["risk_score"] >= 0.65

    def test_predict_low_prob_gives_low_risk(self, sample_claim):
        predictor = ClaimPredictor(
            pipeline=_make_mock_pipeline(prob=0.20),
            features=SAMPLE_FEATURES,
            model_name="xgboost",
        )
        result = predictor.predict(sample_claim)
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] < 0.40

    def test_predict_handles_missing_features(self, mock_predictor, incomplete_claim):
        """Incomplete claim dict must not crash — missing features → NaN → imputed."""
        result = mock_predictor.predict(incomplete_claim)
        assert "risk_score" in result
        assert result["features_received"] < result["features_expected"]

    def test_predict_no_claim_id(self, mock_predictor):
        """Claim dict without claim_id must not crash."""
        claim_no_id = {
            "proc_no_diag": True,
            "diagnosis_code_missing": True,
        }
        result = mock_predictor.predict(claim_no_id)
        assert result["claim_id"] is None

    def test_predict_batch_returns_list(self, mock_predictor, sample_claim):
        pipeline = _make_mock_pipeline(prob=0.75)
        pipeline.predict_proba.return_value = np.array([
            [0.25, 0.75], [0.60, 0.40]
        ])
        predictor = ClaimPredictor(
            pipeline=pipeline,
            features=SAMPLE_FEATURES,
            model_name="xgboost",
        )
        results = predictor.predict_batch([sample_claim, sample_claim])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all("risk_score" in r for r in results)

    def test_predict_batch_empty_input(self, mock_predictor):
        results = mock_predictor.predict_batch([])
        assert results == []

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_train.py"):
            ClaimPredictor.load(models_dir=tmp_path, model_name="xgboost")

    def test_load_raises_on_unknown_model_name(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown model_name"):
            ClaimPredictor.load(models_dir=tmp_path, model_name="random_forest")

    def test_load_from_pkl(self, tmp_path, sample_claim):
        """Save and reload a predictor — predict must still work."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        import numpy as np, pickle

        # Build a real (minimal) pipeline that can be pickled
        X_fake = np.random.rand(20, len(SAMPLE_FEATURES))
        y_fake = np.array([0]*10 + [1]*10)
        real_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   LogisticRegression(max_iter=50)),
        ])
        real_pipeline.fit(X_fake, y_fake)

        pkl_path = tmp_path / "xgb_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"pipeline": real_pipeline, "features": SAMPLE_FEATURES}, f)

        loaded = ClaimPredictor.load(models_dir=tmp_path, model_name="xgboost")
        result = loaded.predict(sample_claim)
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


# ── SHAP explainer tests ──────────────────────────────────────────────────────

class TestSHAPExplainer:
    def test_feature_labels_cover_all_features(self):
        """Every ML feature should have a human-readable label."""
        for feat in SAMPLE_FEATURES:
            assert feat in FEATURE_LABELS, f"Missing label for feature: {feat}"

    def test_fix_suggestions_cover_all_features(self):
        """Every ML feature should have a fix suggestion."""
        for feat in SAMPLE_FEATURES:
            assert feat in FIX_SUGGESTIONS, f"Missing fix for feature: {feat}"

    def test_explain_returns_top_reasons(self, sample_claim):
        """Test explain() structure using a mock explainer."""
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {
            "imputer": MagicMock(),
            "model":   MagicMock(),
        }
        mock_pipeline.named_steps["imputer"].transform.return_value = (
            np.zeros((1, len(SAMPLE_FEATURES)))
        )

        mock_shap_vals = np.array([
            [0.31, 0.24, 0.18, 0.10, 0.05, 0.02, 0.01, 0.005,
             -0.01, -0.02, -0.03, -0.04, -0.05]
        ])

        with patch("shap.TreeExplainer") as mock_tree_exp:
            instance = mock_tree_exp.return_value
            instance.shap_values.return_value = mock_shap_vals  # 2D: (n_samples, n_features)
            instance.expected_value = 0.1

            explainer = SHAPExplainer(
                pipeline=mock_pipeline,
                feature_names=SAMPLE_FEATURES,
            )
            result = explainer.explain(sample_claim, top_n=3)

        assert "top_reasons" in result
        assert len(result["top_reasons"]) == 3
        for reason in result["top_reasons"]:
            assert "rank" in reason
            assert "feature" in reason
            assert "label" in reason
            assert "shap_value" in reason
            assert "direction" in reason
            assert "fix" in reason
            assert reason["direction"] in {"increases_risk", "decreases_risk"}

    def test_explain_reasons_sorted_by_magnitude(self, sample_claim):
        """Top reasons must be sorted by |SHAP value| descending."""
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {
            "imputer": MagicMock(),
            "model":   MagicMock(),
        }
        mock_pipeline.named_steps["imputer"].transform.return_value = (
            np.zeros((1, len(SAMPLE_FEATURES)))
        )

        # SHAP values with clear magnitude ordering
        shap_vals = np.zeros(len(SAMPLE_FEATURES))
        shap_vals[0] = 0.50  # diagnosis_code_missing (largest)
        shap_vals[1] = 0.30  # procedure_code_missing
        shap_vals[2] = -0.20 # billed_amount_missing (abs = 0.20)

        with patch("shap.TreeExplainer") as mock_tree_exp:
            instance = mock_tree_exp.return_value
            instance.shap_values.return_value = shap_vals.reshape(1, -1)
            instance.expected_value = 0.0

            explainer = SHAPExplainer(
                pipeline=mock_pipeline,
                feature_names=SAMPLE_FEATURES,
            )
            result = explainer.explain(sample_claim, top_n=3)

        magnitudes = [abs(r["shap_value"]) for r in result["top_reasons"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_direction_positive_increases_risk(self, sample_claim):
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {
            "imputer": MagicMock(), "model": MagicMock()
        }
        mock_pipeline.named_steps["imputer"].transform.return_value = (
            np.zeros((1, len(SAMPLE_FEATURES)))
        )
        shap_vals = np.zeros(len(SAMPLE_FEATURES))
        shap_vals[0] = 0.5  # positive → increases risk

        with patch("shap.TreeExplainer") as mock_tree_exp:
            instance = mock_tree_exp.return_value
            instance.shap_values.return_value = shap_vals.reshape(1, -1)
            instance.expected_value = 0.0

            explainer = SHAPExplainer(pipeline=mock_pipeline, feature_names=SAMPLE_FEATURES)
            result = explainer.explain(sample_claim, top_n=1)

        assert result["top_reasons"][0]["direction"] == "increases_risk"
