"""
Pytest Configuration & Shared Fixtures
========================================
Loaded automatically by pytest before any test module.

Provides
--------
- Logging setup (WARNING level to keep test output clean)
- Sample DataFrames matching the real CSV structure for each dataset
- Fixtures are scoped to function by default (isolated per test)
"""

import pandas as pd
import pytest

from src.config import setup_logging


# ── Logging ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure structlog once before the test session starts."""
    setup_logging(level="WARNING")


# ── Sample DataFrames ─────────────────────────────────────────────────────────
# These mirror the real CSV structure including intentional nulls and edge cases.
# Tests must use these fixtures — never depend on real data files existing.


@pytest.fixture
def sample_claims_df() -> pd.DataFrame:
    """Minimal claims DataFrame with nulls matching real data patterns."""
    return pd.DataFrame(
        {
            "claim_id":       ["C0001", "C0002", "C0003", "C0004"],
            "patient_id":     ["P001",  "P002",  "P003",  "P004"],
            "provider_id":    ["PR100", "PR101", "PR102", "PR100"],
            "diagnosis_code": ["D10",   None,    "D20",   None],     # 50% null
            "procedure_code": ["PROC1", "PROC2", None,    None],     # 50% null
            "billed_amount":  [5000.0,  None,    12000.0, None],     # 50% null
            "date":           ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )


@pytest.fixture
def sample_providers_df() -> pd.DataFrame:
    """Minimal providers DataFrame with a null location (mirrors real data)."""
    return pd.DataFrame(
        {
            "provider_id": ["PR100", "PR101", "PR102"],
            "doctor_name": ["Dr Patel", "Dr Singh", "Dr Khan"],
            "specialty":   ["Cardiology", "Neurology", "Orthopedic"],
            "location":    ["Mumbai", None, "Bangalore"],  # 1 null
        }
    )


@pytest.fixture
def sample_diagnosis_df() -> pd.DataFrame:
    """Complete diagnosis reference table."""
    return pd.DataFrame(
        {
            "diagnosis_code": ["D10", "D20", "D30", "D40", "D50"],
            "category":       ["Heart", "Bone", "Fever", "Skin", "Diabetes"],
            "severity":       ["High", "High", "Low", "Low", "High"],
        }
    )


@pytest.fixture
def sample_cost_df() -> pd.DataFrame:
    """Complete cost benchmark reference table."""
    return pd.DataFrame(
        {
            "procedure_code": ["PROC1", "PROC2", "PROC3"],
            "average_cost":   [4000,    12000,   7000],
            "expected_cost":  [5000,    15000,   9000],
            "region":         ["Delhi", "Mumbai", "Bangalore"],
        }
    )


@pytest.fixture
def all_sample_dfs(
    sample_claims_df,
    sample_providers_df,
    sample_diagnosis_df,
    sample_cost_df,
) -> dict[str, pd.DataFrame]:
    """Convenience fixture bundling all four sample DataFrames."""
    return {
        "claims":    sample_claims_df,
        "providers": sample_providers_df,
        "diagnosis": sample_diagnosis_df,
        "cost":      sample_cost_df,
    }


@pytest.fixture
def csv_raw_dir(tmp_path, all_sample_dfs) -> "Path":
    """
    Write all sample DataFrames as CSVs to a temp directory.
    Returns the raw directory path.
    Used by integration tests that exercise the full pipeline.
    """
    from src.ingestion.ingest import DATASET_REGISTRY

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    filename_map = {name: filename for name, filename in DATASET_REGISTRY.items()}
    for name, df in all_sample_dfs.items():
        df.to_csv(raw_dir / filename_map[name], index=False)

    return raw_dir
