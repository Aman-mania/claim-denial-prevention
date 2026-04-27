"""
Bronze Layer — Ingestion Pipeline
===================================
Reads raw CSV files → validates schema → attaches metadata → writes Parquet.

Bronze contract (never violate these)
--------------------------------------
1. NO transformations — raw data is preserved byte-for-byte in column values.
2. NO rows dropped — even shell records (all-null critical fields) stay.
3. NO imputation — nulls stay null.
4. Two columns added ONLY: ingestion_timestamp, source_file.
5. Output is immutable Parquet — overwrite only, never mutate in place.

Extending in later weeks
-------------------------
- Add a new source: add one entry to DATASET_REGISTRY in this file.
- Silver pipeline imports DATASET_REGISTRY to know what Bronze files exist.
- Schema changes: update schema.py only — pipeline picks them up automatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera as pa
import structlog

from src.ingestion.schema import SCHEMA_REGISTRY

logger = structlog.get_logger(__name__)


# ── Dataset registry ──────────────────────────────────────────────────────────
# Maps logical dataset name → raw CSV filename.
# Schema lookup is done via SCHEMA_REGISTRY (schema.py).
# To add a new data source: add one entry here + one schema in schema.py.

DATASET_REGISTRY: dict[str, str] = {
    "claims":    "claims_1000.csv",
    "providers": "providers_1000.csv",
    "diagnosis": "diagnosis.csv",
    "cost":      "cost.csv",
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class BronzeIngestionPipeline:
    """
    Orchestrates ingestion of raw CSVs into the Bronze Parquet layer.

    Parameters
    ----------
    raw_dir    : Path to directory containing source CSV files.
    bronze_dir : Path to output root directory for Bronze Parquet files.

    Output layout
    -------------
    bronze_dir/
        claims/claims_bronze.parquet
        providers/providers_bronze.parquet
        diagnosis/diagnosis_bronze.parquet
        cost/cost_bronze.parquet
    """

    def __init__(self, raw_dir: Path, bronze_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.bronze_dir = Path(bronze_dir)
        # Single timestamp for the entire run — all rows in a run share it.
        self._run_ts: str = datetime.now(timezone.utc).isoformat()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_csv(self, dataset_name: str, filename: str) -> pd.DataFrame:
        """Load a CSV from raw_dir. Raises FileNotFoundError if missing."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Raw CSV not found: {path}. "
                f"Place source files in '{self.raw_dir}' before running ingestion."
            )
        df = pd.read_csv(path)
        logger.info(
            "csv_loaded",
            dataset=dataset_name,
            rows=len(df),
            columns=list(df.columns),
            path=str(path),
        )
        return df

    def _attach_metadata(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """
        Attach Bronze provenance columns.
        These are the ONLY columns added to raw data in Bronze.
        """
        df = df.copy()
        df["ingestion_timestamp"] = self._run_ts
        df["source_file"] = source_file
        return df

    def _validate_schema(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> dict:
        """
        Soft schema validation — warns but never rejects data.

        Bronze preserves everything. Schema failures are logged as warnings
        so Silver can decide what to do with bad rows. We never silently drop
        data because of a schema mismatch in Bronze.

        Returns a result dict: {"status": "passed"|"warnings", "errors": str|None}
        """
        schema = SCHEMA_REGISTRY.get(dataset_name)
        if schema is None:
            logger.warning("no_schema_registered", dataset=dataset_name)
            return {"status": "skipped", "errors": None}

        # Validate only raw columns — strip the two metadata columns we added.
        meta_cols = {"ingestion_timestamp", "source_file"}
        raw_cols = [c for c in df.columns if c not in meta_cols]
        df_to_validate = df[raw_cols]

        try:
            schema.validate(df_to_validate, lazy=True)
            logger.info("schema_validation_passed", dataset=dataset_name)
            return {"status": "passed", "errors": None}
        except pa.errors.SchemaErrors as exc:
            # Collect all failures and surface them as a warning — not an exception.
            failure_cases = exc.failure_cases.to_dict(orient="records")
            logger.warning(
                "schema_validation_warnings",
                dataset=dataset_name,
                failure_count=len(failure_cases),
                sample_failures=failure_cases[:3],  # log first 3 only
            )
            return {
                "status": "warnings",
                "errors": failure_cases,
            }

    def _write_parquet(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """Write DataFrame to Bronze Parquet. Creates directory if needed."""
        out_dir = self.bronze_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_bronze.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            "bronze_written",
            dataset=dataset_name,
            rows=len(df),
            path=str(out_path),
        )
        return out_path

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, datasets: Optional[list[str]] = None) -> dict:
        """
        Run the Bronze ingestion pipeline.

        Parameters
        ----------
        datasets : List of dataset names to process.
                   None (default) processes all datasets in DATASET_REGISTRY.

        Returns
        -------
        run_report : dict with structure:
            {
                "run_timestamp": str,
                "datasets": {
                    "<name>": {
                        "status":      "success" | "failed",
                        "raw_rows":    int,
                        "output_path": str,
                        "validation":  {"status": str, "errors": list | None},
                    }
                }
            }
        """
        targets = datasets if datasets is not None else list(DATASET_REGISTRY.keys())
        run_report: dict = {
            "run_timestamp": self._run_ts,
            "datasets": {},
        }

        for name in targets:
            if name not in DATASET_REGISTRY:
                logger.error("unknown_dataset", dataset=name, known=list(DATASET_REGISTRY.keys()))
                run_report["datasets"][name] = {
                    "status": "failed",
                    "error": f"'{name}' not in DATASET_REGISTRY",
                }
                continue

            filename = DATASET_REGISTRY[name]
            logger.info("ingestion_start", dataset=name, source_file=filename)

            try:
                df = self._load_csv(name, filename)
                raw_rows = len(df)

                df = self._attach_metadata(df, source_file=filename)
                validation = self._validate_schema(df, name)

                out_path = self._write_parquet(df, name)

                run_report["datasets"][name] = {
                    "status":      "success",
                    "raw_rows":    raw_rows,
                    "output_path": str(out_path),
                    "validation":  validation,
                }
                logger.info("ingestion_success", dataset=name, rows=raw_rows)

            except FileNotFoundError as exc:
                logger.error("ingestion_failed", dataset=name, error=str(exc))
                run_report["datasets"][name] = {"status": "failed", "error": str(exc)}

            except Exception as exc:
                logger.exception("ingestion_unexpected_error", dataset=name, error=str(exc))
                run_report["datasets"][name] = {"status": "failed", "error": str(exc)}

        return run_report
