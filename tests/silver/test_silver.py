"""
Tests — Silver Cleaning Pipeline
===================================
Unit tests for each cleaning function plus integration tests
for the full pipeline run.

Covers:
  - Date parsing
  - Missing-value flags
  - Text standardisation
  - Deduplication
  - Negative amount handling
  - Edge cases: empty DataFrames, all-null columns, no duplicates
"""

import pandas as pd
import pytest

from src.silver.clean import SilverCleaningPipeline
from src.silver.schema import SILVER_SCHEMA_REGISTRY


# ── Fixture: pipeline instance without file I/O ───────────────────────────────

@pytest.fixture
def pipeline(tmp_path):
    """Pipeline with dummy paths — used for unit tests of pure cleaning methods."""
    return SilverCleaningPipeline(
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
    )


# ── _clean_claims — unit tests ─────────────────────────────────────────────────

class TestCleanClaims:
    def test_date_parsed_to_datetime(self, pipeline, sample_claims_df):
        result = pipeline._clean_claims(sample_claims_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_unparseable_date_becomes_nat(self, pipeline):
        df = pd.DataFrame({
            "claim_id":       ["C001"],
            "patient_id":     ["P001"],
            "provider_id":    ["PR100"],
            "diagnosis_code": ["D10"],
            "procedure_code": ["PROC1"],
            "billed_amount":  [5000.0],
            "date":           ["not-a-date"],  # should become NaT
        })
        result = pipeline._clean_claims(df)
        assert pd.isna(result["date"].iloc[0])

    def test_missing_flags_added(self, pipeline, sample_claims_df):
        result = pipeline._clean_claims(sample_claims_df)
        assert "diagnosis_code_missing"  in result.columns
        assert "procedure_code_missing"  in result.columns
        assert "billed_amount_missing"   in result.columns

    def test_missing_flags_are_bool_dtype(self, pipeline, sample_claims_df):
        result = pipeline._clean_claims(sample_claims_df)
        assert result["diagnosis_code_missing"].dtype == bool
        assert result["procedure_code_missing"].dtype == bool
        assert result["billed_amount_missing"].dtype == bool

    def test_flag_matches_sentinel_fill(self, pipeline, sample_claims_df):
        """Flag True where sentinel fill happened (codes) or where amount is still null."""
        result = pipeline._clean_claims(sample_claims_df)
        # Code flags: True ↔ value is "MISSING" (original was null)
        assert (result["diagnosis_code_missing"] == (result["diagnosis_code"] == "MISSING")).all()
        assert (result["procedure_code_missing"] == (result["procedure_code"] == "MISSING")).all()
        # Amount flag: True ↔ still null (never filled)
        assert (result["billed_amount_missing"] == result["billed_amount"].isnull()).all()

    def test_string_codes_filled_with_sentinel(self, pipeline, sample_claims_df):
        """Silver fills null string codes with 'MISSING' so downstream never gets null strings."""
        result = pipeline._clean_claims(sample_claims_df)
        # No nulls remain in code columns after Silver
        assert result["diagnosis_code"].isnull().sum() == 0
        assert result["procedure_code"].isnull().sum() == 0
        # Sentinel value is "MISSING"
        assert (result["diagnosis_code"] == "MISSING").sum() == sample_claims_df["diagnosis_code"].isnull().sum()

    def test_billed_amount_never_filled(self, pipeline, sample_claims_df):
        """billed_amount is intentionally never filled — financial data stays null."""
        before = sample_claims_df["billed_amount"].isnull().sum()
        result = pipeline._clean_claims(sample_claims_df)
        assert result["billed_amount"].isnull().sum() == before

    def test_diagnosis_code_uppercased_and_stripped(self, pipeline):
        df = pd.DataFrame({
            "claim_id":       ["C001"],
            "patient_id":     ["P001"],
            "provider_id":    ["PR100"],
            "diagnosis_code": ["  d10  "],  # lower + spaces
            "procedure_code": [" proc1 "],
            "billed_amount":  [5000.0],
            "date":           ["2024-01-01"],
        })
        result = pipeline._clean_claims(df)
        assert result["diagnosis_code"].iloc[0] == "D10"
        assert result["procedure_code"].iloc[0] == "PROC1"

    def test_negative_billed_amount_nullified(self, pipeline):
        df = pd.DataFrame({
            "claim_id":       ["C001"],
            "patient_id":     ["P001"],
            "provider_id":    ["PR100"],
            "diagnosis_code": ["D10"],
            "procedure_code": ["PROC1"],
            "billed_amount":  [-500.0],
            "date":           ["2024-01-01"],
        })
        result = pipeline._clean_claims(df)
        assert pd.isna(result["billed_amount"].iloc[0])
        # Flag must be True now (was set to null)
        assert result["billed_amount_missing"].iloc[0] == True

    def test_valid_billed_amount_unchanged(self, pipeline):
        df = pd.DataFrame({
            "claim_id":       ["C001"],
            "patient_id":     ["P001"],
            "provider_id":    ["PR100"],
            "diagnosis_code": ["D10"],
            "procedure_code": ["PROC1"],
            "billed_amount":  [5000.0],
            "date":           ["2024-01-01"],
        })
        result = pipeline._clean_claims(df)
        assert result["billed_amount"].iloc[0] == 5000.0

    def test_duplicate_claim_ids_removed(self, pipeline):
        df = pd.DataFrame({
            "claim_id":       ["C001", "C001", "C002"],
            "patient_id":     ["P001", "P001", "P002"],
            "provider_id":    ["PR100", "PR100", "PR101"],
            "diagnosis_code": ["D10",  "D10",  "D20"],
            "procedure_code": ["PROC1", "PROC1", "PROC2"],
            "billed_amount":  [5000.0, 5000.0, 12000.0],
            "date":           ["2024-01-01", "2024-01-01", "2024-01-02"],
        })
        result = pipeline._clean_claims(df)
        assert len(result) == 2
        assert result["claim_id"].duplicated().sum() == 0

    def test_no_duplicate_rows_unchanged(self, pipeline, sample_claims_df):
        result = pipeline._clean_claims(sample_claims_df)
        # sample_claims_df has no duplicates — row count must not change
        assert len(result) == len(sample_claims_df)

    def test_silver_timestamp_added(self, pipeline, sample_claims_df):
        result = pipeline._clean_claims(sample_claims_df)
        assert "silver_timestamp" in result.columns
        assert result["silver_timestamp"].notna().all()

    def test_does_not_mutate_input(self, pipeline, sample_claims_df):
        original_cols = list(sample_claims_df.columns)
        pipeline._clean_claims(sample_claims_df)
        assert list(sample_claims_df.columns) == original_cols


# ── _clean_providers — unit tests ─────────────────────────────────────────────

class TestCleanProviders:
    def test_location_missing_flag_added(self, pipeline, sample_providers_df):
        result = pipeline._clean_providers(sample_providers_df)
        assert "location_missing" in result.columns

    def test_location_missing_true_for_nulls(self, pipeline, sample_providers_df):
        result = pipeline._clean_providers(sample_providers_df)
        null_before = sample_providers_df["location"].isnull()
        assert (result["location_missing"] == null_before).all()

    def test_specialty_title_cased(self, pipeline):
        df = pd.DataFrame({
            "provider_id": ["PR100"],
            "doctor_name": ["Dr Patel"],
            "specialty":   ["  cardiology  "],
            "location":    ["Mumbai"],
        })
        result = pipeline._clean_providers(df)
        assert result["specialty"].iloc[0] == "Cardiology"

    def test_location_title_cased(self, pipeline):
        df = pd.DataFrame({
            "provider_id": ["PR100"],
            "doctor_name": ["Dr Patel"],
            "specialty":   ["Cardiology"],
            "location":    ["  mumbai  "],
        })
        result = pipeline._clean_providers(df)
        assert result["location"].iloc[0] == "Mumbai"

    def test_duplicate_provider_ids_removed(self, pipeline):
        df = pd.DataFrame({
            "provider_id": ["PR100", "PR100"],
            "doctor_name": ["Dr Patel", "Dr Patel"],
            "specialty":   ["Cardiology", "Cardiology"],
            "location":    ["Mumbai", "Mumbai"],
        })
        result = pipeline._clean_providers(df)
        assert len(result) == 1


# ── _clean_diagnosis — unit tests ─────────────────────────────────────────────

class TestCleanDiagnosis:
    def test_code_uppercased(self, pipeline, sample_diagnosis_df):
        df = sample_diagnosis_df.copy()
        df["diagnosis_code"] = df["diagnosis_code"].str.lower()
        result = pipeline._clean_diagnosis(df)
        assert result["diagnosis_code"].str.isupper().all()

    def test_category_title_cased(self, pipeline, sample_diagnosis_df):
        df = sample_diagnosis_df.copy()
        df["category"] = df["category"].str.lower()
        result = pipeline._clean_diagnosis(df)
        # Title case: "heart" → "Heart"
        assert all(c[0].isupper() for c in result["category"])


# ── _clean_cost — unit tests ───────────────────────────────────────────────────

class TestCleanCost:
    def test_cost_ratio_added(self, pipeline, sample_cost_df):
        result = pipeline._clean_cost(sample_cost_df)
        assert "cost_ratio" in result.columns

    def test_cost_ratio_value(self, pipeline, sample_cost_df):
        result = pipeline._clean_cost(sample_cost_df)
        # PROC1: average=4000, expected=5000 → ratio=0.8
        proc1 = result[result["procedure_code"] == "PROC1"]
        if not proc1.empty:
            assert abs(proc1.iloc[0]["cost_ratio"] - 0.8) < 0.001

    def test_procedure_code_uppercased(self, pipeline):
        df = pd.DataFrame({
            "procedure_code": ["proc1"],
            "average_cost":   [4000],
            "expected_cost":  [5000],
            "region":         ["delhi"],
        })
        result = pipeline._clean_cost(df)
        assert result["procedure_code"].iloc[0] == "PROC1"
        assert result["region"].iloc[0] == "Delhi"

    def test_numeric_coercion(self, pipeline):
        """Non-numeric cost values should become NaN, not crash."""
        df = pd.DataFrame({
            "procedure_code": ["PROC1"],
            "average_cost":   ["4000"],   # stored as string
            "expected_cost":  ["5000"],
            "region":         ["Delhi"],
        })
        result = pipeline._clean_cost(df)
        assert result["average_cost"].iloc[0] == 4000.0


# ── Full pipeline integration tests ───────────────────────────────────────────


    def test_proc_no_diag_flag(self, pipeline):
        """proc_no_diag = True when procedure is present but diagnosis is MISSING sentinel."""
        df = pd.DataFrame({
            "claim_id":       ["C001", "C002", "C003"],
            "patient_id":     ["P001", "P002", "P003"],
            "provider_id":    ["PR100", "PR101", "PR102"],
            "diagnosis_code": [None,   "D10",   None],    # C001 no diag, C002 has diag, C003 no diag
            "procedure_code": ["PROC1", "PROC2", None],   # C001 has proc, C002 has proc, C003 no proc
            "billed_amount":  [5000.0, 12000.0, None],
            "date":           ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        result = pipeline._clean_claims(df)
        # C001: procedure=PROC1, diagnosis=MISSING → proc_no_diag should be True
        assert result.loc[result["claim_id"] == "C001", "proc_no_diag"].iloc[0] == True
        # C002: both present → proc_no_diag should be False
        assert result.loc[result["claim_id"] == "C002", "proc_no_diag"].iloc[0] == False
        # C003: procedure=MISSING, diagnosis=MISSING → proc_no_diag should be False (no procedure)
        assert result.loc[result["claim_id"] == "C003", "proc_no_diag"].iloc[0] == False

    def test_diag_no_proc_flag(self, pipeline):
        """diag_no_proc = True when diagnosis is present but procedure is MISSING sentinel."""
        df = pd.DataFrame({
            "claim_id":       ["C001", "C002"],
            "patient_id":     ["P001", "P002"],
            "provider_id":    ["PR100", "PR101"],
            "diagnosis_code": ["D10",  "D20"],    # both have diagnosis
            "procedure_code": [None,   "PROC1"],  # C001 missing proc, C002 has proc
            "billed_amount":  [5000.0, 12000.0],
            "date":           ["2024-01-01", "2024-01-02"],
        })
        result = pipeline._clean_claims(df)
        assert result.loc[result["claim_id"] == "C001", "diag_no_proc"].iloc[0] == True
        assert result.loc[result["claim_id"] == "C002", "diag_no_proc"].iloc[0] == False

    def test_new_flags_are_bool(self, pipeline, sample_claims_df):
        """proc_no_diag and diag_no_proc must be boolean dtype."""
        result = pipeline._clean_claims(sample_claims_df)
        assert result["proc_no_diag"].dtype == bool
        assert result["diag_no_proc"].dtype == bool

class TestSilverPipelineIntegration:
    def _setup_bronze(self, tmp_path, all_sample_dfs):
        """Write sample DataFrames as Bronze Parquet files."""
        from src.ingestion.ingest import BronzeIngestionPipeline
        raw_dir    = tmp_path / "raw"
        bronze_dir = tmp_path / "bronze"
        raw_dir.mkdir()

        filename_map = {
            "claims":    "claims_1000.csv",
            "providers": "providers_1000.csv",
            "diagnosis": "diagnosis.csv",
            "cost":      "cost.csv",
        }
        for name, df in all_sample_dfs.items():
            df.to_csv(raw_dir / filename_map[name], index=False)

        ingestion = BronzeIngestionPipeline(raw_dir=raw_dir, bronze_dir=bronze_dir)
        ingestion.run()
        return bronze_dir

    def test_full_run_creates_silver_files(self, tmp_path, all_sample_dfs):
        bronze_dir = self._setup_bronze(tmp_path, all_sample_dfs)
        silver_dir = tmp_path / "silver"
        pipeline   = SilverCleaningPipeline(bronze_dir=bronze_dir, silver_dir=silver_dir)
        report     = pipeline.run()

        for name in ["claims", "providers", "diagnosis", "cost"]:
            assert report["datasets"][name]["status"] == "success"
            expected = silver_dir / name / f"{name}_silver.parquet"
            assert expected.exists()

    def test_silver_row_count_lte_bronze(self, tmp_path, all_sample_dfs):
        """Silver rows <= Bronze rows (dedup may remove, never add)."""
        bronze_dir = self._setup_bronze(tmp_path, all_sample_dfs)
        silver_dir = tmp_path / "silver"
        pipeline   = SilverCleaningPipeline(bronze_dir=bronze_dir, silver_dir=silver_dir)
        report     = pipeline.run()

        for name in ["claims", "providers", "diagnosis", "cost"]:
            d = report["datasets"][name]
            assert d["silver_rows"] <= d["bronze_rows"]

    def test_silver_claims_has_flag_columns(self, tmp_path, all_sample_dfs):
        bronze_dir = self._setup_bronze(tmp_path, all_sample_dfs)
        silver_dir = tmp_path / "silver"
        pipeline   = SilverCleaningPipeline(bronze_dir=bronze_dir, silver_dir=silver_dir)
        pipeline.run(datasets=["claims"])

        df = pd.read_parquet(silver_dir / "claims" / "claims_silver.parquet")
        assert "diagnosis_code_missing"  in df.columns
        assert "procedure_code_missing"  in df.columns
        assert "billed_amount_missing"   in df.columns

    def test_silver_cost_has_cost_ratio(self, tmp_path, all_sample_dfs):
        bronze_dir = self._setup_bronze(tmp_path, all_sample_dfs)
        silver_dir = tmp_path / "silver"
        pipeline   = SilverCleaningPipeline(bronze_dir=bronze_dir, silver_dir=silver_dir)
        pipeline.run(datasets=["cost"])

        df = pd.read_parquet(silver_dir / "cost" / "cost_silver.parquet")
        assert "cost_ratio" in df.columns

    def test_missing_bronze_fails_gracefully(self, tmp_path):
        """Pipeline should return failed status, not raise unhandled exception."""
        pipeline = SilverCleaningPipeline(
            bronze_dir=tmp_path / "nonexistent",
            silver_dir=tmp_path / "silver",
        )
        report = pipeline.run(datasets=["claims"])
        assert report["datasets"]["claims"]["status"] == "failed"

    def test_partial_run(self, tmp_path, all_sample_dfs):
        bronze_dir = self._setup_bronze(tmp_path, all_sample_dfs)
        silver_dir = tmp_path / "silver"
        pipeline   = SilverCleaningPipeline(bronze_dir=bronze_dir, silver_dir=silver_dir)
        report     = pipeline.run(datasets=["claims"])

        assert "claims"    in report["datasets"]
        assert "providers" not in report["datasets"]


# ── Silver schema tests ────────────────────────────────────────────────────────

class TestSilverSchema:
    def test_all_datasets_have_schemas(self):
        for name in ["claims", "providers", "diagnosis", "cost"]:
            assert name in SILVER_SCHEMA_REGISTRY

    def test_claims_schema_passes_valid_silver_data(self, pipeline, sample_claims_df):
        silver = pipeline._clean_claims(sample_claims_df)
        # Strip date and metadata cols before schema validation
        skip = {"date", "silver_timestamp", "ingestion_timestamp", "source_file"}
        validate_cols = [c for c in silver.columns if c not in skip]
        SILVER_SCHEMA_REGISTRY["claims"].validate(silver[validate_cols], lazy=True)

    def test_cost_schema_passes_valid_silver_data(self, pipeline, sample_cost_df):
        silver = pipeline._clean_cost(sample_cost_df)
        skip   = {"silver_timestamp", "ingestion_timestamp", "source_file"}
        validate_cols = [c for c in silver.columns if c not in skip]
        SILVER_SCHEMA_REGISTRY["cost"].validate(silver[validate_cols], lazy=True)
