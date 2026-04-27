"""
Silver Layer — Cleaning Pipeline
===================================
Transforms Bronze Parquet → clean, trusted Silver Parquet.

Silver contract (never violate)
---------------------------------
1. NO rows dropped — problem rows are FLAGGED with boolean columns.
2. NO cost imputation — billed_amount nulls stay null.
3. Row count in Silver <= Bronze (only strict duplicates on ID are removed).
4. Every cleaning step is a pure function (df in → df out, no side effects).
5. Flag columns (e.g. diagnosis_code_missing) carry forward into Gold features.

Missing value strategy (per column)
--------------------------------------
  diagnosis_code  → fill null with "MISSING", add diagnosis_code_missing flag
  procedure_code  → fill null with "MISSING", add procedure_code_missing flag
  billed_amount   → keep null (NEVER impute cost), add billed_amount_missing flag
  date            → parse to datetime; unparseable rows get NaT
  location        → fill null with "Unknown", add location_missing flag

Business logic flags added in Silver (set on sentinel-filled data):
  proc_no_diag    → procedure present but diagnosis absent (billing without justification)
  diag_no_proc    → diagnosis present but procedure absent (condition documented, nothing billed)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa
import structlog

from src.silver.schema import SILVER_SCHEMA_REGISTRY

logger = structlog.get_logger(__name__)

_META_COLS = frozenset({"ingestion_timestamp", "source_file"})


class SilverCleaningPipeline:
    """
    Reads Bronze Parquet → applies per-dataset cleaning → writes Silver Parquet.

    Parameters
    ----------
    bronze_dir : Root of Bronze Parquet subdirectories.
    silver_dir : Root of Silver Parquet subdirectories (created if absent).
    """

    def __init__(self, bronze_dir: Path, silver_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self._run_ts   = datetime.now(timezone.utc).isoformat()

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load_bronze(self, dataset_name: str) -> pd.DataFrame:
        path = self.bronze_dir / dataset_name / f"{dataset_name}_bronze.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Bronze file not found: {path}. Run ingestion (run_ingestion.py) first."
            )
        df = pd.read_parquet(path)
        logger.info("bronze_loaded", dataset=dataset_name, rows=len(df))
        return df

    def _write_silver(self, df: pd.DataFrame, dataset_name: str) -> Path:
        out_dir = self.silver_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_silver.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("silver_written", dataset=dataset_name, rows=len(df), path=str(out_path))
        return out_path

    # ── Per-dataset cleaning — pure functions ─────────────────────────────────

    def _clean_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Claims cleaning steps:
          1. Parse date string → datetime (NaT where unparseable)
          2. Strip + uppercase code columns
          3. Nullify negative billed_amount (invalid)
          4. Add boolean missing-value flags for 3 critical fields
          5. Remove strict claim_id duplicates (keep first)
          6. Add business logic violation flags (proc_no_diag, diag_no_proc)
          7. Attach silver_timestamp
        """
        df = df.copy()

        # 1. Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 2. Standardise code columns: strip whitespace, uppercase
        for col in ["diagnosis_code", "procedure_code"]:
            df[col] = df[col].str.strip().str.upper()

        # 3. Negative billed_amount → null (business rule: cost can't be negative)
        invalid_amount = df["billed_amount"].notna() & (df["billed_amount"] < 0)
        if invalid_amount.sum() > 0:
            logger.warning("negative_billed_amount_nullified", count=int(invalid_amount.sum()))
            df.loc[invalid_amount, "billed_amount"] = None

        # 4. Flags captured BEFORE sentinel fill — flags reflect original raw nulls
        df["diagnosis_code_missing"] = df["diagnosis_code"].isnull()
        df["procedure_code_missing"] = df["procedure_code"].isnull()
        df["billed_amount_missing"]  = df["billed_amount"].isnull()

        # 5. Sentinel fill for string code columns
        #    "MISSING" replaces null codes so downstream code never gets null strings.
        #    The flag columns above are the source of truth for ML features.
        #    billed_amount is intentionally NOT filled — never impute financial data.
        df["diagnosis_code"] = df["diagnosis_code"].fillna("MISSING")
        df["procedure_code"] = df["procedure_code"].fillna("MISSING")

        # 7. Dedup by claim_id — keep first occurrence
        dupe_mask = df.duplicated(subset=["claim_id"], keep="first")
        if dupe_mask.sum() > 0:
            logger.warning("duplicate_claim_ids_removed", count=int(dupe_mask.sum()))
        df = df[~dupe_mask].reset_index(drop=True)

        # 8. Business logic violation flags
        #    Set AFTER sentinel fill so "MISSING" codes are treated as absent.
        #    proc_no_diag: procedure billed without a diagnosis code — primary denial trigger
        #    diag_no_proc: diagnosis documented but no procedure billed — incomplete claim
        df["proc_no_diag"] = (df["procedure_code"] != "MISSING") & (df["diagnosis_code"] == "MISSING")
        df["diag_no_proc"] = (df["diagnosis_code"] != "MISSING") & (df["procedure_code"] == "MISSING")

        # 9. Silver metadata
        df["silver_timestamp"] = self._run_ts

        return df

    def _clean_providers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Providers cleaning steps:
          1. Strip whitespace from all text columns
          2. Title-case specialty and location
          3. Add location_missing flag
          4. Deduplicate by provider_id
        """
        df = df.copy()

        # Strip whitespace
        for col in ["doctor_name", "specialty", "location"]:
            if col in df.columns:
                df[col] = df[col].str.strip()

        # Title-case location and specialty for consistency
        df["specialty"] = df["specialty"].str.title()
        # Flag missing location BEFORE fill, then sentinel fill
        df["location_missing"] = df["location"].isnull()
        df["location"] = df["location"].fillna("Unknown").str.title()

        # Deduplicate by provider_id
        dupe_mask = df.duplicated(subset=["provider_id"], keep="first")
        if dupe_mask.sum() > 0:
            logger.warning("duplicate_provider_ids_removed", count=int(dupe_mask.sum()))
        df = df[~dupe_mask].reset_index(drop=True)

        df["silver_timestamp"] = self._run_ts
        return df

    def _clean_diagnosis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Diagnosis cleaning:
          1. Uppercase diagnosis_code
          2. Title-case category and severity
        Reference table is already clean — minimal transformation needed.
        """
        df = df.copy()
        df["diagnosis_code"] = df["diagnosis_code"].str.strip().str.upper()
        df["category"]       = df["category"].str.strip().str.title()
        df["severity"]       = df["severity"].str.strip().str.title()
        df["silver_timestamp"] = self._run_ts
        return df

    def _clean_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cost cleaning:
          1. Uppercase procedure_code, title-case region
          2. Ensure numeric types for cost columns
          3. Add cost_ratio = average_cost / expected_cost (useful for Gold features)
        """
        df = df.copy()
        df["procedure_code"] = df["procedure_code"].str.strip().str.upper()
        df["region"]         = df["region"].str.strip().str.title()

        # Ensure float types (safe coercion)
        df["average_cost"]  = pd.to_numeric(df["average_cost"],  errors="coerce").astype("float64")
        df["expected_cost"] = pd.to_numeric(df["expected_cost"], errors="coerce").astype("float64")

        # Cost ratio: how does actual average compare to expected?
        df["cost_ratio"] = (df["average_cost"] / df["expected_cost"]).round(4)

        df["silver_timestamp"] = self._run_ts
        return df

    # ── Schema validation (soft — warns, never rejects) ────────────────────────

    def _validate(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Validate cleaned DataFrame against Silver schema.
        Soft: logs warnings but never raises. Bronze already has the raw data.
        Strips pipeline metadata before validating.
        """
        schema = SILVER_SCHEMA_REGISTRY.get(dataset_name)
        if schema is None:
            return {"status": "skipped", "errors": None}

        # Exclude metadata and silver_timestamp columns from schema check
        skip = _META_COLS | {"silver_timestamp", "date"}
        validate_cols = [c for c in df.columns if c not in skip]
        df_check = df[validate_cols]

        try:
            schema.validate(df_check, lazy=True)
            logger.info("silver_validation_passed", dataset=dataset_name)
            return {"status": "passed", "errors": None}
        except pa.errors.SchemaErrors as exc:
            failures = exc.failure_cases.to_dict(orient="records")
            logger.warning(
                "silver_validation_warnings",
                dataset=dataset_name,
                failure_count=len(failures),
                sample=failures[:3],
            )
            return {"status": "warnings", "errors": failures}
        except Exception as exc:
            logger.error("silver_validation_unexpected", dataset=dataset_name, error=str(exc))
            return {"status": "error", "errors": str(exc)}

    # ── Orchestration ──────────────────────────────────────────────────────────

    # Maps dataset name → cleaning function
    _CLEANERS: dict = {
        "claims":    "_clean_claims",
        "providers": "_clean_providers",
        "diagnosis": "_clean_diagnosis",
        "cost":      "_clean_cost",
    }

    def run(self, datasets: list[str] | None = None) -> dict:
        """
        Run the Silver cleaning pipeline for one or all datasets.

        Parameters
        ----------
        datasets : Subset to process. None = all known datasets.

        Returns
        -------
        dict with run_timestamp and per-dataset results including
        bronze_rows, silver_rows, rows_removed, validation status.
        """
        targets = datasets or list(self._CLEANERS.keys())
        report: dict = {"run_timestamp": self._run_ts, "datasets": {}}

        for name in targets:
            if name not in self._CLEANERS:
                logger.error("unknown_dataset_silver", dataset=name)
                report["datasets"][name] = {"status": "failed", "error": f"Unknown dataset: {name}"}
                continue

            logger.info("silver_cleaning_start", dataset=name)
            cleaner = getattr(self, self._CLEANERS[name])

            try:
                df_bronze   = self._load_bronze(name)
                bronze_rows = len(df_bronze)

                df_silver   = cleaner(df_bronze)
                silver_rows = len(df_silver)

                validation  = self._validate(df_silver, name)
                out_path    = self._write_silver(df_silver, name)

                report["datasets"][name] = {
                    "status":       "success",
                    "bronze_rows":  bronze_rows,
                    "silver_rows":  silver_rows,
                    "rows_removed": bronze_rows - silver_rows,
                    "output_path":  str(out_path),
                    "validation":   validation,
                }
                logger.info(
                    "silver_cleaning_complete",
                    dataset=name,
                    bronze_rows=bronze_rows,
                    silver_rows=silver_rows,
                    rows_removed=bronze_rows - silver_rows,
                )

            except FileNotFoundError as exc:
                logger.error("silver_cleaning_failed", dataset=name, error=str(exc))
                report["datasets"][name] = {"status": "failed", "error": str(exc)}

            except Exception as exc:
                logger.exception("silver_cleaning_unexpected_error", dataset=name, error=str(exc))
                report["datasets"][name] = {"status": "failed", "error": str(exc)}

        return report
