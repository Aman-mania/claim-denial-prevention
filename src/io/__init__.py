"""IO abstractions for local-first and cloud-ready table access."""

from src.io.table_store import LocalTableStore, TableStore, table_store_from_env

__all__ = ["LocalTableStore", "TableStore", "table_store_from_env"]
