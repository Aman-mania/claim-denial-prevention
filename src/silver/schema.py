"""
Silver Layer — Pandera Schemas
================================
Stricter than Bronze schemas. Key differences:
  - Boolean flag columns (e.g. diagnosis_code_missing) are validated
  - billed_amount remains nullable (NEVER impute costs)
  - date column is excluded from schema — its datetime dtype is
    verified in the cleaning tests directly (pandas version variance)

Import pattern:
    from src.silver.schema import SILVER_CLAIMS_SCHEMA
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ── Claims ────────────────────────────────────────────────────────────────────

SILVER_CLAIMS_SCHEMA = DataFrameSchema(
    columns={
        "claim_id":                Column(str,   nullable=False),
        "patient_id":              Column(str,   nullable=False),
        "provider_id":             Column(str,   nullable=False),
        "diagnosis_code":          Column(str,   nullable=False),  # filled with "MISSING" sentinel
        "procedure_code":          Column(str,   nullable=False),  # filled with "MISSING" sentinel
        "billed_amount":           Column(float, nullable=True,
                                          checks=Check.ge(0, error="billed_amount must be >= 0")),
        # Missing-field flags
        "diagnosis_code_missing":  Column(bool,  nullable=False),
        "procedure_code_missing":  Column(bool,  nullable=False),
        "billed_amount_missing":   Column(bool,  nullable=False),
        # Business logic violation flags — set after sentinel fill
        "proc_no_diag":            Column(bool,  nullable=False),  # procedure w/o diagnosis
        "diag_no_proc":            Column(bool,  nullable=False),  # diagnosis w/o procedure
    },
    name="claims_silver",
    description="Cleaned claims — null flags added, duplicates removed",
)


# ── Providers ─────────────────────────────────────────────────────────────────

SILVER_PROVIDERS_SCHEMA = DataFrameSchema(
    columns={
        "provider_id":      Column(str,  nullable=False),
        "doctor_name":      Column(str,  nullable=False),
        "specialty":        Column(str,  nullable=False),
        "location":         Column(str,  nullable=False),  # filled with "Unknown" sentinel
        "location_missing": Column(bool, nullable=False),
    },
    name="providers_silver",
    description="Cleaned providers — location flag added, text standardised",
)


# ── Diagnosis ─────────────────────────────────────────────────────────────────

SILVER_DIAGNOSIS_SCHEMA = DataFrameSchema(
    columns={
        "diagnosis_code": Column(str, nullable=False),
        "category":       Column(str, nullable=False),
        "severity":       Column(
            str,
            nullable=False,
            checks=Check.isin(["High", "Low"], error="severity must be 'High' or 'Low'"),
        ),
    },
    name="diagnosis_silver",
    description="Cleaned diagnosis reference — text standardised",
)


# ── Cost ──────────────────────────────────────────────────────────────────────

SILVER_COST_SCHEMA = DataFrameSchema(
    columns={
        "procedure_code": Column(str,   nullable=False),
        "average_cost":   Column(float, nullable=False, checks=Check.ge(0)),
        "expected_cost":  Column(float, nullable=False, checks=Check.ge(0)),
        "region":         Column(str,   nullable=False),
        "cost_ratio":     Column(float, nullable=True),  # derived — may be null if division fails
    },
    name="cost_silver",
    description="Cleaned cost benchmarks — cost_ratio added",
)


# ── Registry ──────────────────────────────────────────────────────────────────

SILVER_SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "claims":    SILVER_CLAIMS_SCHEMA,
    "providers": SILVER_PROVIDERS_SCHEMA,
    "diagnosis": SILVER_DIAGNOSIS_SCHEMA,
    "cost":      SILVER_COST_SCHEMA,
}
