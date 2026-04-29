"""
Tests — Gold Feature Pipeline
================================
Tests for feature engineering, synthetic label creation, and Gold table correctness.
All tests use sample Silver DataFrames — no real Parquet files required.
"""

import numpy as np
import pandas as pd
import pytest

from src.gold.features import GoldFeaturePipeline, SEVERITY_MAP
from src.constants import (
    SENTINEL_MISSING, COL_DIAG_MISSING, COL_PROC_MISSING,
    COL_AMOUNT_MISSING, COL_PROC_NO_DIAG, COL_DIAG_NO_PROC,
)


@pytest.fixture
def pipeline(tmp_path):
    return GoldFeaturePipeline(
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
    )


@pytest.fixture
def silver_claims():
    # C006: all 5 flags True → raw score=0.90, guaranteed denial regardless of noise
    return pd.DataFrame({
        "claim_id":                 ["C001", "C002", "C003", "C004", "C005", "C006"],
        "patient_id":               ["P001", "P001", "P002", "P003", "P003", "P004"],
        "provider_id":              ["PR100", "PR101", "PR100", "PR101", "PR100", "PR101"],
        "diagnosis_code":           ["D10", SENTINEL_MISSING, "D20", "D10", SENTINEL_MISSING, SENTINEL_MISSING],
        "procedure_code":           ["PROC1", "PROC2", SENTINEL_MISSING, "PROC1", SENTINEL_MISSING, SENTINEL_MISSING],
        "billed_amount":            [5000.0, 20000.0, 12000.0, None, None, None],
        "date":                     pd.to_datetime(["2024-01-01", "2024-01-02",
                                                     "2024-01-03", "2024-01-04",
                                                     "2024-01-05", "2024-01-06"]),
        COL_DIAG_MISSING:           [False, True,  False, False, True,  True],
        COL_PROC_MISSING:           [False, False, True,  False, True,  True],
        COL_AMOUNT_MISSING:         [False, False, False, True,  True,  True],
        COL_PROC_NO_DIAG:           [False, True,  False, False, False, True],
        COL_DIAG_NO_PROC:           [False, False, True,  False, False, False],
    })


@pytest.fixture
def silver_providers():
    return pd.DataFrame({
        "provider_id":    ["PR100", "PR101"],
        "specialty":      ["Cardiology", "Neurology"],
        "location":       ["Mumbai", "Unknown"],
        "location_missing": [False, True],
    })


@pytest.fixture
def silver_diagnosis():
    return pd.DataFrame({
        "diagnosis_code": ["D10", "D20", SENTINEL_MISSING],
        "category":       ["Heart", "Bone", "Unknown"],
        "severity":       ["High", "High", "Low"],
    })


@pytest.fixture
def silver_cost():
    return pd.DataFrame({
        "procedure_code": ["PROC1", "PROC2"],
        "expected_cost":  [5000.0, 15000.0],
        "average_cost":   [4000.0, 12000.0],
        "cost_ratio":     [0.8, 0.8],
    })


# ── Base table tests ──────────────────────────────────────────────────────────

class TestBuildBase:
    def test_row_count_preserved(self, pipeline, silver_claims, silver_providers,
                                  silver_diagnosis, silver_cost):
        base = pipeline._build_base(silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost)
        assert len(base) == len(silver_claims)

    def test_specialty_joined(self, pipeline, silver_claims, silver_providers,
                               silver_diagnosis, silver_cost):
        base = pipeline._build_base(silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost)
        assert "specialty" in base.columns
        # All rows have a provider → specialty should be non-null for all
        assert base["specialty"].notna().all()

    def test_deviation_only_when_both_present(self, pipeline, silver_claims,
                                               silver_providers, silver_diagnosis,
                                               silver_cost):
        """billed_deviation_pct must be null when either billed_amount or expected_cost is null."""
        base = pipeline._build_base(silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost)
        # C004: billed_amount is null → deviation must be null
        c004 = base[base["claim_id"] == "C004"]
        assert pd.isna(c004["billed_deviation_pct"].iloc[0])

    def test_deviation_computed_correctly(self, pipeline, silver_claims,
                                           silver_providers, silver_diagnosis,
                                           silver_cost):
        base = pipeline._build_base(silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost)
        # C001: billed=5000, expected=5000 → deviation=0%
        c001 = base[base["claim_id"] == "C001"]
        assert abs(c001["billed_deviation_pct"].iloc[0] - 0.0) < 0.01

    def test_no_rows_dropped_on_left_join(self, pipeline, silver_claims,
                                           silver_providers, silver_diagnosis,
                                           silver_cost):
        """Left joins must never drop rows even when join keys are MISSING sentinel."""
        base = pipeline._build_base(silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost)
        assert set(base["claim_id"]) == set(silver_claims["claim_id"])


# ── Synthetic label tests ─────────────────────────────────────────────────────

class TestDenialLabel:
    def _make_base(self, pipeline, claims, providers, diagnosis, cost):
        base = pipeline._build_base(claims, providers, diagnosis, cost)
        return pipeline._create_denial_label(base)

    def test_denial_flag_is_binary(self, pipeline, silver_claims, silver_providers,
                                    silver_diagnosis, silver_cost):
        base = self._make_base(pipeline, silver_claims, silver_providers,
                                silver_diagnosis, silver_cost)
        assert set(base["denial_flag"].unique()).issubset({0, 1})

    def test_denial_risk_score_range(self, pipeline, silver_claims, silver_providers,
                                      silver_diagnosis, silver_cost):
        base = self._make_base(pipeline, silver_claims, silver_providers,
                                silver_diagnosis, silver_cost)
        assert base["denial_risk_score"].between(0, 1).all()

    def test_label_is_reproducible(self, pipeline, silver_claims, silver_providers,
                                    silver_diagnosis, silver_cost):
        """Same input must always produce the same label (fixed seed)."""
        base1 = self._make_base(pipeline, silver_claims, silver_providers,
                                 silver_diagnosis, silver_cost)
        base2 = self._make_base(pipeline, silver_claims, silver_providers,
                                 silver_diagnosis, silver_cost)
        assert (base1["denial_flag"] == base2["denial_flag"]).all()

    def test_shell_claims_tend_to_be_denied(self, pipeline, silver_claims,
                                             silver_providers, silver_diagnosis,
                                             silver_cost):
        """Claims with all 3 fields missing should have high denial_risk_score."""
        base = self._make_base(pipeline, silver_claims, silver_providers,
                                silver_diagnosis, silver_cost)
        # C005: diagnosis=MISSING, procedure=MISSING, amount=null
        c005 = base[base["claim_id"] == "C005"]
        # Risk score should be above 0.4 (structural flags trigger weight)
        assert c005["denial_risk_score"].iloc[0] >= 0.3


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestBuildFeatures:
    def _make_features(self, pipeline, claims, providers, diagnosis, cost):
        base = pipeline._build_base(claims, providers, diagnosis, cost)
        base = pipeline._create_denial_label(base)
        return pipeline._build_features(base)

    def test_provider_claim_count_correct(self, pipeline, silver_claims,
                                           silver_providers, silver_diagnosis,
                                           silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        # PR100 appears in C001, C003, C005 → claim_count = 3
        pr100 = features[features["provider_id"] == "PR100"]
        assert (pr100["provider_claim_count"] == 3).all()

    def test_patient_claim_count_correct(self, pipeline, silver_claims,
                                          silver_providers, silver_diagnosis,
                                          silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        # P001 has C001, C002 → 2 claims
        p001 = features[features["patient_id"] == "P001"]
        assert (p001["patient_claim_count"] == 2).all()

    def test_severity_encoded_values(self, pipeline, silver_claims,
                                      silver_providers, silver_diagnosis,
                                      silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        # D10=Heart→High→2, D20=Bone→High→2, MISSING→null→0
        assert set(features["severity_encoded"].unique()).issubset({0, 1, 2})

    def test_severity_high_gets_2(self, pipeline, silver_claims,
                                   silver_providers, silver_diagnosis, silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        c001 = features[features["claim_id"] == "C001"]
        assert c001["severity_encoded"].iloc[0] == 2  # D10 → High

    def test_specialty_encoded_not_null(self, pipeline, silver_claims,
                                         silver_providers, silver_diagnosis,
                                         silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        # specialty always exists (100% coverage) → encoded should never be null
        assert features["specialty_encoded"].notna().all()

    def test_log_billed_amount_null_when_amount_null(self, pipeline, silver_claims,
                                                      silver_providers, silver_diagnosis,
                                                      silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        # C004 and C005 have null billed_amount → log_billed_amount must be null
        null_rows = features[features["billed_amount_missing"] == True]
        assert null_rows["log_billed_amount"].isna().all()

    def test_log_billed_amount_equals_log1p(self, pipeline, silver_claims,
                                             silver_providers, silver_diagnosis,
                                             silver_cost):
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        c001 = features[features["claim_id"] == "C001"]
        expected = round(float(np.log1p(5000.0)), 4)
        actual   = round(float(c001["log_billed_amount"].iloc[0]), 4)
        assert abs(actual - expected) < 0.001

    def test_billed_deviation_capped_at_500(self, pipeline, silver_claims,
                                             silver_providers, silver_diagnosis,
                                             silver_cost):
        """Deviation must be capped at 500% regardless of actual value."""
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        assert features["billed_deviation_capped"].dropna().le(500).all()
        assert features["billed_deviation_capped"].dropna().ge(-100).all()

    def test_row_count_unchanged(self, pipeline, silver_claims, silver_providers,
                                  silver_diagnosis, silver_cost):
        """Feature engineering must not add or drop rows."""
        features = self._make_features(pipeline, silver_claims, silver_providers,
                                        silver_diagnosis, silver_cost)
        assert len(features) == len(silver_claims)


# ── Feature manifest tests ────────────────────────────────────────────────────

class TestFeatureManifest:
    def test_manifest_has_ml_use_flag(self, pipeline):
        manifest = pipeline.build_feature_manifest()
        for item in manifest:
            assert "ml_use" in item
            assert isinstance(item["ml_use"], bool)

    def test_ml_features_have_names(self, pipeline):
        manifest = pipeline.build_feature_manifest()
        ml_features = [m["name"] for m in manifest if m["ml_use"]]
        assert len(ml_features) >= 10  # at least 10 ML features

    def test_denial_flag_not_ml_feature(self, pipeline):
        """denial_flag is the target — must not appear as an ML feature."""
        manifest = pipeline.build_feature_manifest()
        ml_features = [m["name"] for m in manifest if m["ml_use"]]
        assert "denial_flag" not in ml_features

    def test_claim_id_not_ml_feature(self, pipeline):
        manifest = pipeline.build_feature_manifest()
        ml_features = [m["name"] for m in manifest if m["ml_use"]]
        assert "claim_id" not in ml_features


# ── Full pipeline integration test ────────────────────────────────────────────

class TestGoldPipelineIntegration:
    def _write_silver(self, tmp_path, silver_claims, silver_providers,
                       silver_diagnosis, silver_cost):
        silver_dir = tmp_path / "silver"
        for name, df in [
            ("claims", silver_claims), ("providers", silver_providers),
            ("diagnosis", silver_diagnosis), ("cost", silver_cost),
        ]:
            d = silver_dir / name
            d.mkdir(parents=True, exist_ok=True)
            df.to_parquet(d / f"{name}_silver.parquet", index=False)
        return silver_dir

    def test_full_run_creates_files(self, tmp_path, silver_claims, silver_providers,
                                     silver_diagnosis, silver_cost):
        silver_dir = self._write_silver(tmp_path, silver_claims, silver_providers,
                                         silver_diagnosis, silver_cost)
        gold_dir   = tmp_path / "gold"
        pipeline   = GoldFeaturePipeline(silver_dir=silver_dir, gold_dir=gold_dir)
        report     = pipeline.run()

        assert report["status"] == "success"
        assert (gold_dir / "gold_claim_base.parquet").exists()
        assert (gold_dir / "gold_claim_features.parquet").exists()
        assert (gold_dir / "feature_manifest.json").exists()

    def test_full_run_report_has_denial_rate(self, tmp_path, silver_claims,
                                              silver_providers, silver_diagnosis,
                                              silver_cost):
        silver_dir = self._write_silver(tmp_path, silver_claims, silver_providers,
                                         silver_diagnosis, silver_cost)
        pipeline = GoldFeaturePipeline(silver_dir=silver_dir, gold_dir=tmp_path / "gold")
        report   = pipeline.run()

        assert "denial_rate_pct" in report
        assert 0 < report["denial_rate_pct"] < 100
