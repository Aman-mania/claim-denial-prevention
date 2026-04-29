"""
Project-wide constants
========================
Only values that are:
  1. Used in two or more files, AND
  2. Would need to change in all those places simultaneously

Do NOT add things here just because they are numbers.
If a value is a function parameter default, it stays where it is.
If a value only appears in one file, it stays where it is.

Adding a constant: put it here, import it in the files that use it,
delete the inline definition from those files.
"""

# ── Sentinel values ────────────────────────────────────────────────────────────
# Used in: src/silver/clean.py (to fill), src/silver/clean.py (flag logic),
#           dev_dashboard/components/charts.py (display fallback)

# Replaces null string codes in Silver — never use the string directly.
# If this changes, clean.py fill AND flag conditions update automatically.
SENTINEL_MISSING = "MISSING"

# Replaces null location in providers Silver layer.
SENTINEL_UNKNOWN = "Unknown"


# ── Pipeline metadata columns ─────────────────────────────────────────────────
# Columns added by the pipeline itself — excluded from all data quality checks,
# null profiles, cardinality counts, and analytics aggregations.
#
# Used in: src/ingestion/profiler.py, src/analytics/aggregations.py,
#          src/silver/clean.py, dev_dashboard/tabs/clean_data.py
#
# Bronze adds: ingestion_timestamp, source_file
# Silver adds: silver_timestamp (in addition to Bronze columns)

BRONZE_META_COLS: frozenset[str] = frozenset({
    "ingestion_timestamp",
    "source_file",
})

# Silver includes all Bronze metadata plus its own timestamp.
# Use this when working with Silver DataFrames.
SILVER_META_COLS: frozenset[str] = BRONZE_META_COLS | {"silver_timestamp"}


# ── Silver flag column names ───────────────────────────────────────────────────
# Boolean flag columns added by Silver cleaning.
# Used in: src/silver/clean.py, src/silver/schema.py,
#          src/analytics/aggregations.py, dev_dashboard/tabs/clean_data.py
#
# Naming rule: <source_column>_missing for null-fill flags,
#              descriptive name for business logic flags.

# Missing-value flags (True = original value was null before sentinel fill)
COL_DIAG_MISSING   = "diagnosis_code_missing"
COL_PROC_MISSING   = "procedure_code_missing"
COL_AMOUNT_MISSING = "billed_amount_missing"
COL_LOC_MISSING    = "location_missing"

# Business logic violation flags (True = structural problem with the claim)
COL_PROC_NO_DIAG = "proc_no_diag"   # procedure present, diagnosis absent
COL_DIAG_NO_PROC = "diag_no_proc"   # diagnosis present, procedure absent

# Convenience tuple for iterating over all Silver claims flags
SILVER_CLAIMS_FLAG_COLS: tuple[str, ...] = (
    COL_DIAG_MISSING,
    COL_PROC_MISSING,
    COL_AMOUNT_MISSING,
    COL_PROC_NO_DIAG,
    COL_DIAG_NO_PROC,
)

# The three critical completeness fields — used for shell claim detection
CRITICAL_FIELDS: tuple[str, ...] = (
    "diagnosis_code",
    "procedure_code",
    "billed_amount",
)


# ── Dashboard ──────────────────────────────────────────────────────────────────
# Used in: dev_dashboard/tabs/raw_data.py, dev_dashboard/tabs/clean_data.py

# How long Streamlit caches loaded Parquet data before re-reading from disk.
DASHBOARD_CACHE_TTL: int = 300  # seconds
