
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "dev_dashboard"
for path in [str(ROOT), str(DASHBOARD)]:
    if path not in sys.path:
        sys.path.insert(0, path)


from tabs import retrieval_analytics as ra  # noqa: E402


def test_retrieval_renderer_accepts_dashboard_kwargs():
    sig = inspect.signature(ra.render_retrieval_analytics_tab)
    assert "root_dir" in sig.parameters
    assert "gold_dir" in sig.parameters
    assert "models_dir" in sig.parameters


def test_reason_source_flow_aggregates_counts():
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
    assert set(flow.columns) == {"reason", "source", "count"}
    assert flow["count"].sum() == 3


def test_reason_coverage_reports_match_rate():
    explanations = pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "reason_code": ["A", "B"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["A"],
            "policy_chunk_id": ["P1"],
            "source_name": ["policy.md"],
            "similarity_score": [0.88],
        }
    )
    coverage = ra._prepare_reason_coverage(explanations, matches)
    row_a = coverage[coverage["reason_code"] == "A"].iloc[0]
    row_b = coverage[coverage["reason_code"] == "B"].iloc[0]
    assert row_a["match_rate"] == 1.0
    assert row_b["match_rate"] == 0.0


def test_top_chunks_deduplicates_and_enriches_metadata():
    matches = pd.DataFrame(
        {
            "claim_id": ["C1", "C1", "C2"],
            "reason_code": ["A", "A", "B"],
            "policy_chunk_id": ["P1", "P1", "P2"],
            "source_name": ["source.md", "source.md", "other.md"],
            "section_title": ["S1", "S1", "S2"],
            "similarity_score": [0.7, 0.8, 0.9],
        }
    )
    chunks = pd.DataFrame(
        {
            "chunk_id": ["P1", "P2"],
            "source_name": ["source.md", "other.md"],
            "section_title": ["Section 1", "Section 2"],
            "chunk_text": ["Policy text one", "Policy text two"],
        }
    )
    out = ra._prepare_top_chunks(matches, chunks)
    assert not out.empty
    assert "retrieval_count" in out.columns
    assert "snippet" in out.columns


def test_vector_projection_flags_selected_claim():
    vectors = np.eye(4, dtype=np.float32)
    chunks = pd.DataFrame(
        {
            "chunk_id": ["P1", "P2", "P3", "P4"],
            "source_name": ["a.md", "a.md", "b.md", "b.md"],
            "section_title": ["A", "B", "C", "D"],
            "chunk_text": ["one", "two", "three", "four"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["A"],
            "policy_chunk_id": ["P2"],
            "source_name": ["a.md"],
            "similarity_score": [0.99],
        }
    )
    out = ra._prepare_vector_projection(vectors, chunks, matches, selected_claim="C1", method="PCA")
    assert not out.empty
    assert out["selected_claim_match"].sum() == 1


def test_shap_similarity_join():
    explanations = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["A"],
            "shap_value": [1.2],
            "risk_level": ["HIGH"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["A"],
            "policy_chunk_id": ["P1"],
            "similarity_score": [0.77],
        }
    )
    out = ra._prepare_shap_similarity(explanations, matches)
    assert len(out) == 1
    assert out.iloc[0]["similarity_score"] == 0.77
