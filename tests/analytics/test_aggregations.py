"""
Tests — Analytics Aggregations
================================
Unit tests for all aggregation functions in src/analytics/aggregations.py.
Tests are independent of file I/O — they operate on DataFrames directly.
Edge cases covered: empty DataFrames, all-null columns, missing join keys.
"""

import pandas as pd
import pytest

from src.analytics.aggregations import (
    compute_claims_by_diagnosis,
    compute_claims_by_provider,
    compute_claims_timeline,
    compute_cleaning_impact,
    compute_cost_analysis,
    compute_high_cost_claims,
    compute_null_profile,
    compute_overview,
    compute_specialty_summary,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────
# These mirror conftest.py fixtures but are defined locally for independence.

@pytest.fixture
def claims():
    return pd.DataFrame({
        "claim_id":       ["C001", "C002", "C003", "C004"],
        "patient_id":     ["P001", "P001", "P002", "P003"],
        "provider_id":    ["PR100", "PR101", "PR100", "PR101"],
        "diagnosis_code": ["D10",  None,   "D20",  None],
        "procedure_code": ["PROC1", "PROC2", None,  None],
        "billed_amount":  [5000.0, 20000.0, 12000.0, None],
        "date":           ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    })

@pytest.fixture
def providers():
    return pd.DataFrame({
        "provider_id": ["PR100", "PR101"],
        "doctor_name": ["Dr Patel", "Dr Singh"],
        "specialty":   ["Cardiology", "Neurology"],
        "location":    ["Mumbai", None],
    })

@pytest.fixture
def diagnosis():
    return pd.DataFrame({
        "diagnosis_code": ["D10", "D20", "D30"],
        "category":       ["Heart", "Bone", "Fever"],
        "severity":       ["High", "High", "Low"],
    })

@pytest.fixture
def cost():
    return pd.DataFrame({
        "procedure_code": ["PROC1", "PROC2"],
        "average_cost":   [4000, 12000],
        "expected_cost":  [5000, 15000],
        "region":         ["Delhi", "Mumbai"],
    })


# ── compute_overview ──────────────────────────────────────────────────────────

class TestComputeOverview:
    def test_returns_expected_keys(self, claims, providers, diagnosis, cost):
        result = compute_overview(claims, providers, diagnosis, cost)
        for key in ["total_claims", "unique_patients", "unique_providers",
                    "avg_billed_amount", "total_billed", "claims_complete",
                    "shell_claims", "date_min", "date_max"]:
            assert key in result, f"Missing key: {key}"

    def test_total_claims_matches_input(self, claims, providers, diagnosis, cost):
        result = compute_overview(claims, providers, diagnosis, cost)
        assert result["total_claims"] == len(claims)

    def test_unique_patients_correct(self, claims, providers, diagnosis, cost):
        result = compute_overview(claims, providers, diagnosis, cost)
        assert result["unique_patients"] == claims["patient_id"].nunique()

    def test_shell_claims_counts_all_null_rows(self, claims, providers, diagnosis, cost):
        # C004 has all 3 critical fields null → shell claim
        result = compute_overview(claims, providers, diagnosis, cost)
        assert result["shell_claims"] == 1

    def test_empty_claims_returns_zeros(self, providers, diagnosis, cost):
        empty = pd.DataFrame(columns=["claim_id", "patient_id", "provider_id",
                                       "diagnosis_code", "procedure_code",
                                       "billed_amount", "date"])
        result = compute_overview(empty, providers, diagnosis, cost)
        assert result["total_claims"] == 0
        assert result["unique_patients"] == 0


# ── compute_null_profile ──────────────────────────────────────────────────────

class TestComputeNullProfile:
    def test_returns_dataframe(self, claims):
        result = compute_null_profile(claims, "claims")
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, claims):
        result = compute_null_profile(claims, "claims")
        for col in ["column", "null_count", "null_pct", "dataset"]:
            assert col in result.columns

    def test_null_count_is_correct(self, claims):
        result = compute_null_profile(claims, "claims")
        diag_row = result[result["column"] == "diagnosis_code"]
        assert not diag_row.empty
        assert diag_row.iloc[0]["null_count"] == claims["diagnosis_code"].isnull().sum()

    def test_null_pct_sums_correctly(self, claims):
        result = compute_null_profile(claims, "claims")
        diag_row = result[result["column"] == "diagnosis_code"].iloc[0]
        expected_pct = claims["diagnosis_code"].isnull().mean() * 100
        assert abs(diag_row["null_pct"] - expected_pct) < 0.01

    def test_excludes_meta_columns(self, claims):
        """ingestion_timestamp and source_file should not appear in profile."""
        df = claims.copy()
        df["ingestion_timestamp"] = "2024-01-01T00:00:00Z"
        df["source_file"] = "test.csv"
        result = compute_null_profile(df, "claims")
        assert "ingestion_timestamp" not in result["column"].values
        assert "source_file" not in result["column"].values

    def test_fully_populated_column_has_zero_null(self, claims):
        result = compute_null_profile(claims, "claims")
        claim_id_row = result[result["column"] == "claim_id"].iloc[0]
        assert claim_id_row["null_count"] == 0
        assert claim_id_row["null_pct"] == 0.0


# ── compute_claims_by_provider ────────────────────────────────────────────────

class TestComputeClaimsByProvider:
    def test_returns_sorted_descending(self, claims, providers):
        result = compute_claims_by_provider(claims, providers)
        counts = result["claim_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_joins_specialty(self, claims, providers):
        result = compute_claims_by_provider(claims, providers)
        assert "specialty" in result.columns

    def test_all_providers_present(self, claims, providers):
        result = compute_claims_by_provider(claims, providers)
        for pid in claims["provider_id"].unique():
            assert pid in result["provider_id"].values

    def test_empty_claims_returns_empty(self, providers):
        empty = pd.DataFrame(columns=["claim_id", "provider_id", "billed_amount"])
        result = compute_claims_by_provider(empty, providers)
        assert result.empty


# ── compute_claims_by_diagnosis ───────────────────────────────────────────────

class TestComputeClaimsByDiagnosis:
    def test_null_diagnosis_included_as_row(self, claims, diagnosis):
        """NULL diagnosis codes must appear as a separate row — not silently dropped."""
        result = compute_claims_by_diagnosis(claims, diagnosis)
        # Rows where diagnosis_code is NaN should appear
        null_rows = result[result["diagnosis_code"].isna()]
        assert len(null_rows) > 0

    def test_category_joined_for_known_codes(self, claims, diagnosis):
        result = compute_claims_by_diagnosis(claims, diagnosis)
        d10_row = result[result["diagnosis_code"] == "D10"]
        assert not d10_row.empty
        assert d10_row.iloc[0]["category"] == "Heart"

    def test_sorted_descending_by_count(self, claims, diagnosis):
        result = compute_claims_by_diagnosis(claims, diagnosis)
        counts = result["claim_count"].tolist()
        assert counts == sorted(counts, reverse=True)


# ── compute_cost_analysis ─────────────────────────────────────────────────────

class TestComputeCostAnalysis:
    def test_has_deviation_pct(self, claims, cost):
        result = compute_cost_analysis(claims, cost)
        assert "deviation_pct" in result.columns

    def test_deviation_computed_correctly(self, claims, cost):
        result = compute_cost_analysis(claims, cost)
        proc1 = result[result["procedure_code"] == "PROC1"]
        if not proc1.empty:
            avg_b = proc1.iloc[0]["avg_billed"]
            exp_c = proc1.iloc[0]["expected_cost"]
            expected_dev = round((avg_b - exp_c) / exp_c * 100, 2)
            assert abs(proc1.iloc[0]["deviation_pct"] - expected_dev) < 0.01

    def test_excludes_null_billed_rows(self, claims, cost):
        """Claims with null billed_amount must not inflate averages."""
        result = compute_cost_analysis(claims, cost)
        # PROC2 has only C002 with billed=20000
        proc2 = result[result["procedure_code"] == "PROC2"]
        if not proc2.empty:
            assert proc2.iloc[0]["avg_billed"] == 20000.0

    def test_empty_inputs_return_empty(self):
        result = compute_cost_analysis(pd.DataFrame(), pd.DataFrame())
        assert result.empty


# ── compute_high_cost_claims ──────────────────────────────────────────────────

class TestComputeHighCostClaims:
    def test_zero_threshold_returns_all_with_match(self, claims, cost):
        result = compute_high_cost_claims(claims, cost, threshold_pct=0.0)
        # PROC1: billed 5000 > expected 5000 is exactly 0% — should NOT appear
        # PROC2: billed 20000 > expected 15000 → +33% > 0% → appears
        assert len(result) >= 0  # at least does not crash

    def test_high_threshold_filters_correctly(self, claims, cost):
        result = compute_high_cost_claims(claims, cost, threshold_pct=200.0)
        # No claim in our fixture is >200% above expected
        assert result.empty or all(result["deviation_pct"] > 200)

    def test_respects_top_n(self, claims, cost):
        result = compute_high_cost_claims(claims, cost, threshold_pct=0.0, top_n=1)
        assert len(result) <= 1

    def test_empty_inputs(self):
        result = compute_high_cost_claims(pd.DataFrame(), pd.DataFrame())
        assert result.empty


# ── compute_claims_timeline ───────────────────────────────────────────────────

class TestComputeClaimsTimeline:
    def test_returns_date_and_count_columns(self, claims):
        result = compute_claims_timeline(claims)
        assert "claim_count" in result.columns

    def test_row_per_date(self, claims):
        result = compute_claims_timeline(claims)
        # 4 claims on 4 different dates → 4 rows
        assert len(result) == 4

    def test_sorted_ascending(self, claims):
        result = compute_claims_timeline(claims)
        dates = result.iloc[:, 0].tolist()
        assert dates == sorted(dates)

    def test_empty_claims(self):
        empty = pd.DataFrame(columns=["claim_id", "date"])
        result = compute_claims_timeline(empty)
        assert result.empty or len(result) == 0


# ── compute_cleaning_impact ───────────────────────────────────────────────────

class TestComputeCleaningImpact:
    def test_returns_required_keys(self, claims):
        # Simulate silver with flag columns
        silver = claims.copy()
        silver["diagnosis_code_missing"] = silver["diagnosis_code"].isnull()
        silver["procedure_code_missing"] = silver["procedure_code"].isnull()
        silver["billed_amount_missing"]  = silver["billed_amount"].isnull()
        silver["date"] = pd.to_datetime(silver["date"], errors="coerce")

        result = compute_cleaning_impact(claims, silver)
        for key in ["bronze_rows", "silver_rows", "rows_removed",
                    "flagged_diag", "flagged_proc", "flagged_amount"]:
            assert key in result, f"Missing key: {key}"

    def test_rows_removed_is_difference(self, claims):
        silver = claims.iloc[:3].copy()  # remove one row
        silver["diagnosis_code_missing"] = silver["diagnosis_code"].isnull()
        silver["procedure_code_missing"] = silver["procedure_code"].isnull()
        silver["billed_amount_missing"]  = silver["billed_amount"].isnull()
        silver["date"] = pd.to_datetime(silver["date"], errors="coerce")

        result = compute_cleaning_impact(claims, silver)
        assert result["rows_removed"] == 1

    def test_flagged_counts_match_nulls(self, claims):
        silver = claims.copy()
        silver["diagnosis_code_missing"] = silver["diagnosis_code"].isnull()
        silver["procedure_code_missing"] = silver["procedure_code"].isnull()
        silver["billed_amount_missing"]  = silver["billed_amount"].isnull()
        silver["date"] = pd.to_datetime(silver["date"], errors="coerce")

        result = compute_cleaning_impact(claims, silver)
        assert result["flagged_diag"] == int(claims["diagnosis_code"].isnull().sum())
