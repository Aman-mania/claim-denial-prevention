"""
Silver Layer — Pandera Schemas
================================
Stricter than Bronze schemas. Key differences:
  - Boolean flag columns are validated.
  - billed_amount remains nullable in Silver; model-ready imputation is built in Gold.
  - denial_flag is optional for backward compatibility but validated when present.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check


SILVER_CLAIMS_SCHEMA = DataFrameSchema(
    columns={
        "claim_id":                Column(str,   nullable=False),
        "patient_id":              Column(str,   nullable=False),
        "provider_id":             Column(str,   nullable=False),
        "diagnosis_code":          Column(str,   nullable=False),
        "procedure_code":          Column(str,   nullable=False),
        "billed_amount":           Column(float, nullable=True,
                                          checks=Check.ge(0, error="billed_amount must be >= 0")),
        "denial_flag":             Column(int,   nullable=False, required=False,
                                          checks=Check.isin([0, 1], error="denial_flag must be 0 or 1")),
        "diagnosis_code_missing":  Column(bool,  nullable=False),
        "procedure_code_missing":  Column(bool,  nullable=False),
        "billed_amount_missing":   Column(bool,  nullable=False),
        "proc_no_diag":            Column(bool,  nullable=False),
        "diag_no_proc":            Column(bool,  nullable=False),
    },
    name="claims_silver",
    description="Cleaned claims — null flags added, duplicates removed",
)


SILVER_PROVIDERS_SCHEMA = DataFrameSchema(
    columns={
        "provider_id":      Column(str,  nullable=False),
        "doctor_name":      Column(str,  nullable=False),
        "specialty":        Column(str,  nullable=False),
        "location":         Column(str,  nullable=False),
        "location_missing": Column(bool, nullable=False),
    },
    name="providers_silver",
    description="Cleaned providers — location flag added, text standardised",
)


SILVER_DIAGNOSIS_SCHEMA = DataFrameSchema(
    columns={
        "diagnosis_code": Column(str, nullable=False),
        "category":       Column(str, nullable=False),
        "severity":       Column(
            str,
            nullable=False,
            checks=Check.isin(["High", "Medium", "Low"],
                              error="severity must be 'High', 'Medium', or 'Low'"),
        ),
    },
    name="diagnosis_silver",
    description="Cleaned diagnosis reference — text standardised",
)


SILVER_COST_SCHEMA = DataFrameSchema(
    columns={
        "procedure_code": Column(str,   nullable=False),
        "average_cost":   Column(float, nullable=False, checks=Check.ge(0)),
        "expected_cost":  Column(float, nullable=False, checks=Check.ge(0)),
        "region":         Column(str,   nullable=False),
        "cost_ratio":     Column(float, nullable=True),
    },
    name="cost_silver",
    description="Cleaned cost benchmarks — cost_ratio added",
)


SILVER_SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "claims":    SILVER_CLAIMS_SCHEMA,
    "providers": SILVER_PROVIDERS_SCHEMA,
    "diagnosis": SILVER_DIAGNOSIS_SCHEMA,
    "cost":      SILVER_COST_SCHEMA,
}
