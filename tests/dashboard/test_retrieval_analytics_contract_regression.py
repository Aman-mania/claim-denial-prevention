from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "dev_dashboard"
for path in [str(ROOT), str(DASHBOARD)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from tabs import retrieval_analytics as ra  # noqa: E402


def test_reason_source_flow_keeps_dataframe_contract():
    matches = pd.DataFrame(
        {
            "claim_id": ["C1", "C1", "C2"],
            "reason_code": ["MISSING_DIAGNOSIS", "MISSING_DIAGNOSIS", "HIGH_COST"],
            "policy_chunk_id": ["P1", "P2", "P2"],
            "source_name": ["medical.md", "coding.md", "coding.md"],
            "similarity_score": [0.8, 0.7, 0.9],
        }
    )
    flow = ra._prepare_reason_source_flow(matches)
    assert list(flow.columns) == ["reason", "source", "count"]
    assert flow["count"].sum() == 3


def test_reason_source_flow_can_return_all_sources_when_requested():
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["MISSING_DIAGNOSIS"],
            "policy_chunk_id": ["P1"],
            "source_name": ["medical.md"],
            "similarity_score": [0.8],
        }
    )
    chunks = pd.DataFrame({"source_name": ["medical.md", "unused_policy.md"]})
    flow, all_sources = ra._prepare_reason_source_flow(matches, chunks, include_all_sources=True)
    assert list(flow.columns) == ["reason", "source", "count"]
    assert set(all_sources) == {"medical.md", "unused_policy.md"}
