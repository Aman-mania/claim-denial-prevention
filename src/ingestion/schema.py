"""
Bronze Layer — Pandera Schemas
================================
Defines expected structure for all four raw datasets.

Design intent
-------------
- Bronze schemas validate STRUCTURE only (columns, types, nullable flags).
- Value constraint checks are intentionally minimal — Bronze preserves raw messiness.
- Unique / referential integrity checks belong in Silver (Week 3).
- Adding a new dataset = add one schema + one entry in DATASET_REGISTRY (ingest.py).

Extending in later weeks
------------------------
- Silver schemas will import these and apply stricter transformations.
- Gold schemas will validate engineered feature columns built on top of Silver.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ── Claims ────────────────────────────────────────────────────────────────────
# Main operational table. Three critical fields have high null rates (known).
# billed_amount >= 0 when present — negative billing is always wrong.

CLAIMS_SCHEMA = DataFrameSchema(
    columns={
        "claim_id":       Column(str,   nullable=False),
        "patient_id":     Column(str,   nullable=False),
        "provider_id":    Column(str,   nullable=False),
        "diagnosis_code": Column(str,   nullable=True),   # 30.7% null in source
        "procedure_code": Column(str,   nullable=True),   # 24.1% null in source
        "billed_amount":  Column(float, nullable=True,    # 34.3% null in source
                                 checks=Check.ge(0, error="billed_amount must be >= 0")),
        "date":           Column(str,   nullable=False),  # kept as str; parsed in Silver
    },
    name="claims_bronze",
    description="Raw claims data — immutable Bronze copy",
)


# ── Providers ─────────────────────────────────────────────────────────────────
# Reference table for provider metadata.
# location is nullable (4/21 providers missing it in source data).

PROVIDERS_SCHEMA = DataFrameSchema(
    columns={
        "provider_id":  Column(str, nullable=False),
        "doctor_name":  Column(str, nullable=False),
        "specialty":    Column(str, nullable=False),
        "location":     Column(str, nullable=True),  # 19% null in source
    },
    name="providers_bronze",
    description="Raw provider reference data — immutable Bronze copy",
)


# ── Diagnosis ─────────────────────────────────────────────────────────────────
# Small reference table. severity must be High or Low — no other values in source.

DIAGNOSIS_SCHEMA = DataFrameSchema(
    columns={
        "diagnosis_code": Column(str, nullable=False),
        "category":       Column(str, nullable=False),
        "severity":       Column(
            str,
            nullable=False,
            checks=Check.isin(
                ["High", "Low"],
                error="severity must be 'High' or 'Low'",
            ),
        ),
    },
    name="diagnosis_bronze",
    description="Raw diagnosis reference data — immutable Bronze copy",
)


# ── Cost ──────────────────────────────────────────────────────────────────────
# Regional cost benchmarks. All numeric values are integers in source (no nulls).
# Note: region is procedure-scoped, not claim-scoped — join logic lives in Silver.

COST_SCHEMA = DataFrameSchema(
    columns={
        "procedure_code": Column(str, nullable=False),
        "average_cost":   Column(int, nullable=False,
                                  checks=Check.ge(0, error="average_cost must be >= 0")),
        "expected_cost":  Column(int, nullable=False,
                                  checks=Check.ge(0, error="expected_cost must be >= 0")),
        "region":         Column(str, nullable=False),
    },
    name="cost_bronze",
    description="Raw regional cost benchmark data — immutable Bronze copy",
)


# ── Registry ──────────────────────────────────────────────────────────────────
# Single source of truth mapping dataset name → its schema.
# Used by both the ingestion pipeline and tests.

SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "claims":    CLAIMS_SCHEMA,
    "providers": PROVIDERS_SCHEMA,
    "diagnosis": DIAGNOSIS_SCHEMA,
    "cost":      COST_SCHEMA,
}
