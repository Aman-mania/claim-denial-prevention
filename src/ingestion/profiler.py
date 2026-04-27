"""
Bronze Layer — Data Profiler
==============================
Generates structured quality profiles from Bronze Parquet files.

Design intent
-------------
- Returns structured dicts — not print-only, not notebook-only.
- Silver and Gold can consume profile dicts to make transformation decisions.
- `print_report()` is a development aid; the real value is the returned dict.
- ID columns are dataset-specific; passing id_col enables duplicate-by-ID checks.

Extending in later weeks
-------------------------
- Week 3 (Silver): extend with referential integrity checks (provider_id exists, etc.)
- Week 4 (Gold):   extend with feature distribution checks
- Week 5 (ML):     extend with class imbalance and outlier profiling
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Default ID column per dataset — used for duplicate-by-ID detection.
_DEFAULT_ID_COLS: dict[str, str] = {
    "claims":    "claim_id",
    "providers": "provider_id",
    "diagnosis": "diagnosis_code",
    "cost":      "procedure_code",
}

# Bronze metadata columns — excluded from all profile calculations.
_META_COLS = frozenset({"ingestion_timestamp", "source_file"})


class DataProfiler:
    """
    Profiles Bronze Parquet files and returns structured quality reports.

    Parameters
    ----------
    bronze_dir : Root directory containing Bronze Parquet subdirectories.
    """

    def __init__(self, bronze_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_bronze(self, dataset_name: str) -> pd.DataFrame:
        path = self.bronze_dir / dataset_name / f"{dataset_name}_bronze.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Bronze file not found: {path}. Run ingestion first."
            )
        return pd.read_parquet(path)

    def _data_cols(self, df: pd.DataFrame) -> list[str]:
        """Return only non-metadata columns."""
        return [c for c in df.columns if c not in _META_COLS]

    def _null_profile(self, df: pd.DataFrame) -> dict:
        total = len(df)
        cols = self._data_cols(df)
        return {
            col: {
                "null_count": int(df[col].isnull().sum()),
                "null_pct":   round(df[col].isnull().mean() * 100, 2),
            }
            for col in cols
        }

    def _duplicate_profile(self, df: pd.DataFrame, id_col: Optional[str]) -> dict:
        data_df = df[self._data_cols(df)]
        result: dict = {
            "full_row_duplicates": int(data_df.duplicated().sum()),
        }
        if id_col and id_col in df.columns:
            result["id_duplicates"] = int(df[id_col].duplicated().sum())
            result["id_col"] = id_col
        return result

    def _cardinality_profile(self, df: pd.DataFrame) -> dict:
        return {
            col: int(df[col].nunique(dropna=True))
            for col in self._data_cols(df)
        }

    def _numeric_profile(self, df: pd.DataFrame) -> dict:
        cols = self._data_cols(df)
        num_cols = df[cols].select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return {}
        stats = df[num_cols].describe().round(2)
        # Convert to plain dict (numpy types → Python native for JSON safety)
        return {
            col: {k: float(v) for k, v in stats[col].items()}
            for col in num_cols
        }

    def _value_counts_profile(self, df: pd.DataFrame, max_cardinality: int = 10) -> dict:
        """
        Value frequency for low-cardinality string columns.
        Useful for spotting unexpected values (e.g., unknown severity codes).
        Skips columns with > max_cardinality unique values.
        """
        cols = self._data_cols(df)
        str_cols = df[cols].select_dtypes(include=["object", "string"]).columns.tolist()
        result = {}
        for col in str_cols:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= max_cardinality:
                counts = df[col].value_counts(dropna=False).head(max_cardinality)
                result[col] = {
                    str(k) if k is not None else "NULL": int(v)
                    for k, v in counts.items()
                }
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def profile(
        self,
        dataset_name: str,
        id_col: Optional[str] = None,
    ) -> dict:
        """
        Generate a full quality profile for one Bronze dataset.

        Parameters
        ----------
        dataset_name : Name of dataset (matches Bronze subdirectory name).
        id_col       : Column to use for ID-level duplicate check. None skips it.

        Returns
        -------
        profile : Structured dict with keys:
            dataset, row_count, column_count, columns,
            nulls, duplicates, cardinality, numeric_stats, value_counts
        """
        logger.info("profiling_start", dataset=dataset_name)
        df = self._load_bronze(dataset_name)

        effective_id_col = id_col or _DEFAULT_ID_COLS.get(dataset_name)

        profile: dict = {
            "dataset":       dataset_name,
            "row_count":     len(df),
            "column_count":  len(self._data_cols(df)),
            "columns":       self._data_cols(df),
            "nulls":         self._null_profile(df),
            "duplicates":    self._duplicate_profile(df, effective_id_col),
            "cardinality":   self._cardinality_profile(df),
            "numeric_stats": self._numeric_profile(df),
            "value_counts":  self._value_counts_profile(df),
        }

        logger.info(
            "profiling_complete",
            dataset=dataset_name,
            rows=profile["row_count"],
        )
        return profile

    def profile_all(self) -> dict[str, dict]:
        """
        Profile all available Bronze datasets.

        Skips datasets whose Bronze file does not exist yet (logs a warning).

        Returns
        -------
        dict mapping dataset_name → profile dict
        """
        datasets = list(_DEFAULT_ID_COLS.keys())
        results: dict[str, dict] = {}

        for name in datasets:
            try:
                results[name] = self.profile(name)
            except FileNotFoundError:
                logger.warning("profile_skipped_not_found", dataset=name)

        return results

    def print_report(self, profiles: dict[str, dict]) -> None:
        """
        Human-readable console report for development and debugging.
        Not used in production flows — profile dicts are consumed programmatically.
        """
        for name, p in profiles.items():
            print(f"\n{'═' * 62}")
            print(f"  {name.upper():<20} {p['row_count']:>5} rows · {p['column_count']} data columns")
            print(f"{'═' * 62}")

            # Null rates
            print("\n  Null rates:")
            has_nulls = False
            for col, stats in p["nulls"].items():
                if stats["null_pct"] > 0:
                    bar = "█" * min(int(stats["null_pct"] / 4), 25)
                    print(f"    {col:<22}  {stats['null_pct']:>5.1f}%  {bar}")
                    has_nulls = True
            if not has_nulls:
                print("    None — all columns fully populated.")

            # Duplicates
            dupes = p["duplicates"]
            print(f"\n  Duplicates:")
            print(f"    Full-row: {dupes['full_row_duplicates']}")
            if "id_duplicates" in dupes:
                print(f"    By {dupes['id_col']}: {dupes['id_duplicates']}")

            # Cardinality highlights (only surprising ones)
            print(f"\n  Cardinality:")
            for col, n in p["cardinality"].items():
                print(f"    {col:<22}  {n} unique values")

            # Numeric stats (key fields only)
            if p["numeric_stats"]:
                print(f"\n  Numeric stats:")
                for col, stats in p["numeric_stats"].items():
                    print(
                        f"    {col:<22}  "
                        f"min={stats['min']:<10.1f}  "
                        f"mean={stats['mean']:<10.1f}  "
                        f"max={stats['max']:.1f}"
                    )

            # Value counts for low-cardinality columns
            if p["value_counts"]:
                print(f"\n  Value distributions:")
                for col, counts in p["value_counts"].items():
                    vals = "  ".join(f"{k}:{v}" for k, v in list(counts.items())[:5])
                    print(f"    {col:<22}  {vals}")
