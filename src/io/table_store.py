"""
Table storage abstraction for local-first / cloud-ready pipelines.

Local development writes Parquet files with pandas. On Databricks, the same
pipeline boundary can be replaced by a Delta-backed implementation without
changing explainability/RAG business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class TableStore(Protocol):
    """Minimal table IO contract used by pipeline layers."""

    def read_table(self, name: str) -> pd.DataFrame:
        """Read a named table into a pandas DataFrame."""

    def write_table(self, name: str, df: pd.DataFrame) -> Path | str:
        """Write a pandas DataFrame and return the destination."""


@dataclass
class LocalTableStore:
    """
    Local Parquet-backed table store.

    This exists so cloud migration later swaps the IO implementation, not the
    explainability/RAG business logic.
    """

    root_dir: Path
    suffix: str = ".parquet"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe_name = name if name.endswith(self.suffix) else f"{name}{self.suffix}"
        return self.root_dir / safe_name

    def read_table(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Table not found: {path}")
        return pd.read_parquet(path)

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path


def table_store_from_env(root_dir: Path, *, env_key: str = "CLAIM_DENIAL_TABLE_STORE") -> TableStore:
    """
    Factory for table storage.

    Default is local Parquet. The environment switch is intentionally present now
    so Databricks migration later only swaps this factory implementation.
    """
    import os

    store_type = os.getenv(env_key, "local_parquet").strip().lower()
    if store_type in {"local", "local_parquet", "parquet"}:
        return LocalTableStore(root_dir=Path(root_dir))

    if store_type in {"delta", "databricks_delta"}:
        raise NotImplementedError(
            "Delta table store is planned for cloud migration. "
            "Use local_parquet in local development."
        )

    raise ValueError(f"Unknown table store type: {store_type}")
