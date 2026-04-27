"""
Tests — Bronze Ingestion Pipeline
====================================
Covers:
  - Unit tests: individual pipeline methods in isolation
  - Integration tests: full pipeline run (CSV → Bronze Parquet)
  - Profiler tests: profiling output structure and correctness

All tests use tmp_path or csv_raw_dir fixtures — never the real data/raw/ files.
Tests must remain green in a clean checkout with no data files present.
"""

import pandas as pd
import pytest

from src.ingestion.ingest import BronzeIngestionPipeline, DATASET_REGISTRY
from src.ingestion.profiler import DataProfiler
from src.ingestion.schema import SCHEMA_REGISTRY


# ── Schema registry ───────────────────────────────────────────────────────────

class TestSchemaRegistry:
    def test_all_datasets_have_schemas(self):
        """Every dataset in DATASET_REGISTRY must have a Pandera schema."""
        for name in DATASET_REGISTRY:
            assert name in SCHEMA_REGISTRY, (
                f"Dataset '{name}' is in DATASET_REGISTRY but has no schema in SCHEMA_REGISTRY. "
                f"Add it to src/ingestion/schema.py."
            )

    def test_schema_validates_valid_data(self, sample_claims_df):
        """CLAIMS_SCHEMA must pass on structurally correct data."""
        SCHEMA_REGISTRY["claims"].validate(sample_claims_df, lazy=True)

    def test_schema_rejects_negative_billed_amount(self):
        """CLAIMS_SCHEMA must reject negative billed_amount."""
        import pandera as pa

        bad_df = pd.DataFrame(
            {
                "claim_id":       ["C0001"],
                "patient_id":     ["P001"],
                "provider_id":    ["PR100"],
                "diagnosis_code": ["D10"],
                "procedure_code": ["PROC1"],
                "billed_amount":  [-100.0],  # invalid
                "date":           ["2024-01-01"],
            }
        )
        with pytest.raises(pa.errors.SchemaErrors):
            SCHEMA_REGISTRY["claims"].validate(bad_df, lazy=True)

    def test_diagnosis_schema_rejects_unknown_severity(self):
        """DIAGNOSIS_SCHEMA must reject severity values outside High/Low."""
        import pandera as pa

        bad_df = pd.DataFrame(
            {
                "diagnosis_code": ["D99"],
                "category":       ["Unknown"],
                "severity":       ["Medium"],  # not in ["High", "Low"]
            }
        )
        with pytest.raises(pa.errors.SchemaErrors):
            SCHEMA_REGISTRY["diagnosis"].validate(bad_df, lazy=True)


# ── BronzeIngestionPipeline — unit tests ──────────────────────────────────────

class TestBronzeIngestionPipelineUnit:
    def test_load_csv_raises_on_missing_file(self, tmp_path):
        """_load_csv must raise FileNotFoundError for missing files."""
        pipeline = BronzeIngestionPipeline(tmp_path / "raw", tmp_path / "bronze")
        with pytest.raises(FileNotFoundError, match="claims_1000.csv"):
            pipeline._load_csv("claims", "claims_1000.csv")

    def test_attach_metadata_adds_two_columns(self, sample_claims_df):
        """_attach_metadata must add ingestion_timestamp and source_file only."""
        pipeline = BronzeIngestionPipeline(
            raw_dir="data/raw", bronze_dir="data/bronze"
        )
        original_cols = set(sample_claims_df.columns)
        result = pipeline._attach_metadata(sample_claims_df, source_file="test.csv")

        new_cols = set(result.columns) - original_cols
        assert new_cols == {"ingestion_timestamp", "source_file"}

    def test_attach_metadata_source_file_value(self, sample_claims_df):
        """source_file column must contain the exact filename passed in."""
        pipeline = BronzeIngestionPipeline(
            raw_dir="data/raw", bronze_dir="data/bronze"
        )
        result = pipeline._attach_metadata(sample_claims_df, source_file="claims_1000.csv")
        assert (result["source_file"] == "claims_1000.csv").all()

    def test_attach_metadata_does_not_mutate_input(self, sample_claims_df):
        """_attach_metadata must return a copy — not mutate the input DataFrame."""
        pipeline = BronzeIngestionPipeline(
            raw_dir="data/raw", bronze_dir="data/bronze"
        )
        original_cols = list(sample_claims_df.columns)
        pipeline._attach_metadata(sample_claims_df, source_file="test.csv")
        assert list(sample_claims_df.columns) == original_cols

    def test_validate_schema_passes_valid_data(self, sample_claims_df):
        """_validate_schema must return status=passed for valid data."""
        pipeline = BronzeIngestionPipeline(
            raw_dir="data/raw", bronze_dir="data/bronze"
        )
        df_with_meta = pipeline._attach_metadata(sample_claims_df, "test.csv")
        result = pipeline._validate_schema(df_with_meta, "claims")
        assert result["status"] == "passed"

    def test_validate_schema_warns_on_bad_data(self):
        """_validate_schema must return status=warnings (not raise) for invalid data."""
        bad_df = pd.DataFrame(
            {
                "claim_id":       ["C0001"],
                "patient_id":     ["P001"],
                "provider_id":    ["PR100"],
                "diagnosis_code": ["D10"],
                "procedure_code": ["PROC1"],
                "billed_amount":  [-999.0],  # invalid: negative
                "date":           ["2024-01-01"],
            }
        )
        pipeline = BronzeIngestionPipeline(
            raw_dir="data/raw", bronze_dir="data/bronze"
        )
        df_with_meta = pipeline._attach_metadata(bad_df, "test.csv")
        result = pipeline._validate_schema(df_with_meta, "claims")

        # Soft validation — must warn, never raise
        assert result["status"] == "warnings"
        assert result["errors"] is not None


# ── BronzeIngestionPipeline — integration tests ───────────────────────────────

class TestBronzeIngestionPipelineIntegration:
    def test_full_run_creates_all_parquet_files(self, tmp_path, csv_raw_dir):
        """Full pipeline run must create one Parquet file per dataset."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        report = pipeline.run()

        for name in DATASET_REGISTRY:
            assert report["datasets"][name]["status"] == "success", (
                f"Dataset '{name}' failed: {report['datasets'][name].get('error')}"
            )
            expected_path = bronze_dir / name / f"{name}_bronze.parquet"
            assert expected_path.exists(), f"Bronze file not created: {expected_path}"

    def test_row_count_preserved(self, tmp_path, csv_raw_dir, sample_claims_df):
        """Bronze must contain the exact same row count as the source CSV."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run(datasets=["claims"])

        df_bronze = pd.read_parquet(bronze_dir / "claims" / "claims_bronze.parquet")
        assert len(df_bronze) == len(sample_claims_df)

    def test_nulls_preserved_in_bronze(self, tmp_path, csv_raw_dir, sample_claims_df):
        """Bronze must not impute or drop nulls — raw messy data is preserved."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run(datasets=["claims"])

        df_bronze = pd.read_parquet(bronze_dir / "claims" / "claims_bronze.parquet")
        for col in ["diagnosis_code", "procedure_code", "billed_amount"]:
            expected_nulls = sample_claims_df[col].isnull().sum()
            actual_nulls   = df_bronze[col].isnull().sum()
            assert actual_nulls == expected_nulls, (
                f"Column '{col}': expected {expected_nulls} nulls in Bronze, "
                f"got {actual_nulls}. Bronze must never impute or drop nulls."
            )

    def test_metadata_columns_present_and_correct(self, tmp_path, csv_raw_dir):
        """Bronze Parquet must contain ingestion_timestamp and source_file."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run(datasets=["claims"])

        df = pd.read_parquet(bronze_dir / "claims" / "claims_bronze.parquet")
        assert "ingestion_timestamp" in df.columns
        assert "source_file" in df.columns
        assert (df["source_file"] == "claims_1000.csv").all()

    def test_metadata_timestamp_consistent_within_run(self, tmp_path, csv_raw_dir):
        """All rows within a single run must share the same ingestion_timestamp."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run(datasets=["claims"])

        df = pd.read_parquet(bronze_dir / "claims" / "claims_bronze.parquet")
        assert df["ingestion_timestamp"].nunique() == 1

    def test_partial_run_only_processes_requested(self, tmp_path, csv_raw_dir):
        """run(datasets=[...]) must only process the specified datasets."""
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        report = pipeline.run(datasets=["claims"])

        assert "claims" in report["datasets"]
        assert "providers" not in report["datasets"]
        assert not (bronze_dir / "providers").exists()

    def test_run_report_structure(self, tmp_path, csv_raw_dir):
        """run() report must include run_timestamp and per-dataset results."""
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=tmp_path / "bronze")
        report = pipeline.run()

        assert "run_timestamp" in report
        assert "datasets" in report
        for name in DATASET_REGISTRY:
            assert name in report["datasets"]
            ds = report["datasets"][name]
            assert "status" in ds
            assert "raw_rows" in ds
            assert "output_path" in ds
            assert "validation" in ds

    def test_unknown_dataset_in_run(self, tmp_path, csv_raw_dir):
        """Unknown dataset names must result in failed status, not crash."""
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=tmp_path / "bronze")
        report = pipeline.run(datasets=["nonexistent_dataset"])
        assert report["datasets"]["nonexistent_dataset"]["status"] == "failed"


# ── DataProfiler tests ────────────────────────────────────────────────────────

class TestDataProfiler:
    def _run_ingestion(self, csv_raw_dir, bronze_dir):
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run()

    def test_profile_returns_expected_keys(self, tmp_path, csv_raw_dir):
        """profile() must return a dict with all required top-level keys."""
        bronze_dir = tmp_path / "bronze"
        self._run_ingestion(csv_raw_dir, bronze_dir)

        profiler = DataProfiler(bronze_dir=bronze_dir)
        p = profiler.profile("claims")

        for key in ["dataset", "row_count", "column_count", "columns",
                    "nulls", "duplicates", "cardinality",
                    "numeric_stats", "value_counts"]:
            assert key in p, f"Missing profile key: '{key}'"

    def test_profile_row_count_matches_bronze(self, tmp_path, csv_raw_dir, sample_claims_df):
        """row_count in profile must match actual Bronze Parquet row count."""
        bronze_dir = tmp_path / "bronze"
        self._run_ingestion(csv_raw_dir, bronze_dir)

        profiler = DataProfiler(bronze_dir=bronze_dir)
        p = profiler.profile("claims")
        assert p["row_count"] == len(sample_claims_df)

    def test_profile_detects_nulls(self, tmp_path, csv_raw_dir, sample_claims_df):
        """Null profiling must correctly report null counts for known-null columns."""
        bronze_dir = tmp_path / "bronze"
        self._run_ingestion(csv_raw_dir, bronze_dir)

        profiler = DataProfiler(bronze_dir=bronze_dir)
        p = profiler.profile("claims")

        expected_diag_nulls = int(sample_claims_df["diagnosis_code"].isnull().sum())
        actual_diag_nulls   = p["nulls"]["diagnosis_code"]["null_count"]
        assert actual_diag_nulls == expected_diag_nulls

    def test_profiler_skips_missing_datasets(self, tmp_path, csv_raw_dir):
        """profile_all() must skip datasets whose Bronze file doesn't exist yet."""
        # Only ingest claims — others missing
        bronze_dir = tmp_path / "bronze"
        pipeline = BronzeIngestionPipeline(raw_dir=csv_raw_dir, bronze_dir=bronze_dir)
        pipeline.run(datasets=["claims"])

        profiler = DataProfiler(bronze_dir=bronze_dir)
        profiles = profiler.profile_all()

        assert "claims" in profiles         # ingested — should be profiled
        assert "providers" not in profiles  # not ingested — should be skipped

    def test_profile_missing_bronze_raises(self, tmp_path):
        """profile() must raise FileNotFoundError if Bronze file doesn't exist."""
        profiler = DataProfiler(bronze_dir=tmp_path / "bronze")
        with pytest.raises(FileNotFoundError):
            profiler.profile("claims")
