# Table Contracts + Dtype Audit Patch

## Added

- `src/io/table_contracts.py`
  - Central table dtype contracts for Bronze metadata, Silver, Gold, and Week 5 explanation outputs.
  - `enforce_table_contract(df, table_name)` for low-cost, type-safe Parquet writes.
  - `parquet_write_kwargs()` for consistent compression settings.

- `tools/audit_tables.py`
  - Scans generated Parquet tables under `data/`.
  - Reports row counts, file sizes, dtypes, null %, unique counts, and optimization suggestions.

- `tools/apply_table_contracts_integration.py`
  - Idempotently wires direct Parquet writers in ingestion, silver, and gold pipelines.
  - Week 5+ `LocalTableStore` already enforces table contracts directly.

- `tests/contracts/`
  - Tests for dtype coercion and the audit tool.

## Changed

- `src/io/table_store.py`
  - `LocalTableStore.write_table()` now enforces known table contracts before writing Parquet.

- `src/ingestion/schema.py`
  - Bronze cost schema now accepts `float` for `average_cost` and `expected_cost`.

## How to apply

```bash
unzip claim_denial_table_contracts_and_audit.zip -d /tmp/table_contract_patch
rsync -av /tmp/table_contract_patch/ ./
python tools/apply_table_contracts_integration.py
```

## Verify

```bash
python -m py_compile src/io/table_contracts.py src/io/table_store.py tools/audit_tables.py tools/apply_table_contracts_integration.py
pytest tests/contracts -v
python run_ingestion.py
python run_silver.py
python run_gold.py
python run_explain.py
python tools/audit_tables.py --details --json logs/table_audit.json
```

## Notes

- Bronze remains source-preserving. The patch does not downcast raw business columns in Bronze.
- Silver/Gold/Week 5 outputs are optimized with semantic and storage-safe dtypes.
- Future Week 6 RAG tables should be added to `src/io/table_contracts.py` before writing them.
