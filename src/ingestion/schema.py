"""
Bronze Layer — Pandera Schemas
================================
Defines expected structure for all raw datasets.

Bronze schemas validate STRUCTURE only. Value constraints stay minimal because
Bronze preserves raw messiness for later Silver/Gold stages.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check


CLAIMS_SCHEMA = DataFrameSchema(
    columns={
        "claim_id":       Column(str,   nullable=False),
        "patient_id":     Column(str,   nullable=False),
        "provider_id":    Column(str,   nullable=False),
        "diagnosis_code": Column(str,   nullable=True),
        "procedure_code": Column(str,   nullable=True),
        "billed_amount":  Column(float, nullable=True,
                                 checks=Check.ge(0, error="billed_amount must be >= 0")),
        "date":           Column(str,   nullable=False),
        # New replacement dataset includes this real label. It is optional so
        # the legacy synthetic-label path still works with old/raw test data.
        "denial_flag":    Column(int,   nullable=False, required=False,
                                 checks=Check.isin([0, 1], error="denial_flag must be 0 or 1")),
    },
    name="claims_bronze",
    description="Raw claims data — immutable Bronze copy",
)


PROVIDERS_SCHEMA = DataFrameSchema(
    columns={
        "provider_id":  Column(str, nullable=False),
        "doctor_name":  Column(str, nullable=False),
        "specialty":    Column(str, nullable=False),
        "location":     Column(str, nullable=True),
    },
    name="providers_bronze",
    description="Raw provider reference data — immutable Bronze copy",
)


DIAGNOSIS_SCHEMA = DataFrameSchema(
    columns={
        "diagnosis_code": Column(str, nullable=False),
        "category":       Column(str, nullable=False),
        "severity":       Column(
            str,
            nullable=False,
            checks=Check.isin(
                ["High", "Medium", "Low"],
                error="severity must be 'High', 'Medium', or 'Low'",
            ),
        ),
    },
    name="diagnosis_bronze",
    description="Raw diagnosis reference data — immutable Bronze copy",
)


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


SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "claims":    CLAIMS_SCHEMA,
    "providers": PROVIDERS_SCHEMA,
    "diagnosis": DIAGNOSIS_SCHEMA,
    "cost":      COST_SCHEMA,
}
