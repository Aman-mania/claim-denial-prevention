"""
Silver Layer — Cleaning Pipeline
===================================
Transforms Bronze Parquet → clean, trusted Silver Parquet.

Silver contract
---------------
1. NO rows dropped except strict ID duplicates.
2. NO overwrite of original billed_amount nulls in Silver.
   Gold creates model-ready imputed amount features while preserving this raw field.
3. Every cleaning step is a pure function.
4. Missing/violation flag columns carry forward into Gold features.

Replacement-data support
------------------------
The new claims file includes a real denial_flag. Silver validates and carries it
forward; Gold decides whether to use this real label or synthesize one for legacy data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa
import structlog

from src.silver.schema import SILVER_SCHEMA_REGISTRY
from src.constants import (
    SENTINEL_MISSING, SENTINEL_UNKNOWN,
    SILVER_META_COLS,
    COL_DIAG_MISSING, COL_PROC_MISSING, COL_AMOUNT_MISSING,
    COL_LOC_MISSING, COL_PROC_NO_DIAG, COL_DIAG_NO_PROC,
)

logger = structlog.get_logger(__name__)
_META_COLS = SILVER_META_COLS


class SilverCleaningPipeline:
    def __init__(self, bronze_dir: Path, silver_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self._run_ts   = datetime.now(timezone.utc).isoformat()

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

    def _clean_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["diagnosis_code", "procedure_code"]:
            df[col] = df[col].astype("string").str.strip().str.upper()

        df["billed_amount"] = pd.to_numeric(df["billed_amount"], errors="coerce")

        # Preserve real labels from the replacement dataset, but surface bad values
        # as validation warnings instead of dropping rows in Silver.
        if "denial_flag" in df.columns:
            df["denial_flag"] = pd.to_numeric(df["denial_flag"], errors="coerce")
            invalid_label = df["denial_flag"].notna() & ~df["denial_flag"].isin([0, 1])
            if invalid_label.any():
                logger.warning("invalid_denial_flag_values", count=int(invalid_label.sum()))
            if df["denial_flag"].notna().all() and not invalid_label.any():
                df["denial_flag"] = df["denial_flag"].astype(int)

        invalid_amount = df["billed_amount"].notna() & (df["billed_amount"] < 0)
        if invalid_amount.any():
            logger.warning("negative_billed_amount_nullified", count=int(invalid_amount.sum()))
            df.loc[invalid_amount, "billed_amount"] = None

        # Flags reflect original post-normalization/nullification state.
        df[COL_DIAG_MISSING]   = df["diagnosis_code"].isnull()
        df[COL_PROC_MISSING]   = df["procedure_code"].isnull()
        df[COL_AMOUNT_MISSING] = df["billed_amount"].isnull()

        df["diagnosis_code"] = df["diagnosis_code"].fillna(SENTINEL_MISSING).astype(str)
        df["procedure_code"] = df["procedure_code"].fillna(SENTINEL_MISSING).astype(str)

        dupe_mask = df.duplicated(subset=["claim_id"], keep="first")
        if dupe_mask.any():
            logger.warning("duplicate_claim_ids_removed", count=int(dupe_mask.sum()))
        df = df[~dupe_mask].reset_index(drop=True)

        df[COL_PROC_NO_DIAG] = (
            (df["procedure_code"] != SENTINEL_MISSING)
            & (df["diagnosis_code"] == SENTINEL_MISSING)
        )
        df[COL_DIAG_NO_PROC] = (
            (df["diagnosis_code"] != SENTINEL_MISSING)
            & (df["procedure_code"] == SENTINEL_MISSING)
        )

        df["silver_timestamp"] = self._run_ts
        return df

    def _clean_providers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["doctor_name", "specialty", "location"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        df["specialty"] = df["specialty"].str.title().astype(str)
        df[COL_LOC_MISSING] = df["location"].isnull()
        df["location"] = df["location"].fillna(SENTINEL_UNKNOWN).str.title().astype(str)

        dupe_mask = df.duplicated(subset=["provider_id"], keep="first")
        if dupe_mask.any():
            logger.warning("duplicate_provider_ids_removed", count=int(dupe_mask.sum()))
        df = df[~dupe_mask].reset_index(drop=True)

        df["silver_timestamp"] = self._run_ts
        return df

    def _clean_diagnosis(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["diagnosis_code"] = df["diagnosis_code"].astype("string").str.strip().str.upper().astype(str)
        df["category"]       = df["category"].astype("string").str.strip().str.title().astype(str)
        df["severity"]       = df["severity"].astype("string").str.strip().str.title().astype(str)
        df["silver_timestamp"] = self._run_ts
        return df

    def _clean_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["procedure_code"] = df["procedure_code"].astype("string").str.strip().str.upper().astype(str)
        df["region"]         = df["region"].astype("string").str.strip().str.title().astype(str)
        df["average_cost"]   = pd.to_numeric(df["average_cost"],  errors="coerce").astype("float64")
        df["expected_cost"]  = pd.to_numeric(df["expected_cost"], errors="coerce").astype("float64")
        df["cost_ratio"]     = (df["average_cost"] / df["expected_cost"]).round(4)
        df["silver_timestamp"] = self._run_ts
        return df

    def _validate(self, df: pd.DataFrame, dataset_name: str) -> dict:
        schema = SILVER_SCHEMA_REGISTRY.get(dataset_name)
        if schema is None:
            return {"status": "skipped", "errors": None}

        skip = _META_COLS | {"date"}
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

    _CLEANERS: dict = {
        "claims":    "_clean_claims",
        "providers": "_clean_providers",
        "diagnosis": "_clean_diagnosis",
        "cost":      "_clean_cost",
    }

    def run(self, datasets: list[str] | None = None) -> dict:
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
            except FileNotFoundError as exc:
                logger.error("silver_cleaning_failed", dataset=name, error=str(exc))
                report["datasets"][name] = {"status": "failed", "error": str(exc)}
            except Exception as exc:
                logger.exception("silver_cleaning_unexpected_error", dataset=name, error=str(exc))
                report["datasets"][name] = {"status": "failed", "error": str(exc)}

        return report
