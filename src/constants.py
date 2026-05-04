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

# Replaces null string codes in Silver — never use the string directly.
SENTINEL_MISSING = "MISSING"

# Replaces null location in providers Silver layer.
SENTINEL_UNKNOWN = "Unknown"


# ── Pipeline metadata columns ─────────────────────────────────────────────────

BRONZE_META_COLS: frozenset[str] = frozenset({
    "ingestion_timestamp",
    "source_file",
})

# Silver includes all Bronze metadata plus its own timestamp.
SILVER_META_COLS: frozenset[str] = BRONZE_META_COLS | {"silver_timestamp"}


# ── Silver flag column names ───────────────────────────────────────────────────

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


# ── Gold feature columns ───────────────────────────────────────────────────────
# These are model-ready columns derived in Gold while preserving original raw
# business fields for auditability and future cloud replay/debugging.

COL_AMOUNT_IMPUTED = "billed_amount_imputed"
COL_AMOUNT_IMPUTATION_STRATEGY = "amount_imputation_strategy"
COL_COST_MATCH_LEVEL = "cost_match_level"
COL_COST_MATCH_ENCODED = "cost_match_encoded"
COL_LABEL_SOURCE = "label_source"


# ── Dashboard ──────────────────────────────────────────────────────────────────

DASHBOARD_CACHE_TTL: int = 300  # seconds
