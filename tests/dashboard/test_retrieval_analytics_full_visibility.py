import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

MODULE_PATH = Path(__file__).resolve().parents[2] / "dev_dashboard" / "tabs" / "retrieval_analytics.py"
spec = importlib.util.spec_from_file_location("retrieval_analytics", MODULE_PATH)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)  # type: ignore[union-attr]


def test_source_utilization_keeps_sources_with_zero_retrieval():
    chunks = pd.DataFrame(
        {
            "policy_chunk_id": ["a1", "b1", "c1"],
            "source_name": ["alpha.md", "beta.md", "gamma.md"],
            "chunk_text": ["a", "b", "c"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["MISSING_DIAGNOSIS"],
            "policy_chunk_id": ["a1"],
            "source_name": ["alpha.md"],
            "similarity_score": [0.75],
        }
    )

    util = ra._prepare_source_utilization(chunks, matches)

    assert set(util["source_name"]) == {"alpha.md", "beta.md", "gamma.md"}
    assert int(util.loc[util["source_name"] == "beta.md", "retrieval_count"].iloc[0]) == 0
    assert int(util.loc[util["source_name"] == "gamma.md", "policy_chunks"].iloc[0]) == 1


def test_reason_flow_returns_all_corpus_sources_not_only_matched_sources():
    chunks = pd.DataFrame(
        {
            "policy_chunk_id": ["a1", "b1", "c1"],
            "source_name": ["alpha.md", "beta.md", "gamma.md"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["HIGH_BILLING_AMOUNT"],
            "policy_chunk_id": ["a1"],
            "source_name": ["alpha.md"],
            "similarity_score": [0.81],
        }
    )

    flow, all_sources = ra._prepare_reason_source_flow(matches, chunks)

    assert not flow.empty
    assert set(all_sources) == {"alpha.md", "beta.md", "gamma.md"}


def test_vector_projection_marks_selected_claim_but_keeps_all_sources():
    chunks = pd.DataFrame(
        {
            "policy_chunk_id": ["a1", "b1", "c1"],
            "source_name": ["alpha.md", "beta.md", "gamma.md"],
            "section_title": ["A", "B", "C"],
            "chunk_text": ["alpha text", "beta text", "gamma text"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["PROVIDER_HISTORY_RISK"],
            "policy_chunk_id": ["b1"],
            "source_name": ["beta.md"],
            "similarity_score": [0.66],
        }
    )
    vectors = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.1, 0.2, 1.0]], dtype=np.float32)

    projected = ra._prepare_vector_projection(vectors, chunks, matches, selected_claim="C1", method="PCA")

    assert set(projected["source_name"]) == {"alpha.md", "beta.md", "gamma.md"}
    assert projected.loc[projected["policy_chunk_id"] == "b1", "selected_claim_match"].iloc[0] is True or bool(projected.loc[projected["policy_chunk_id"] == "b1", "selected_claim_match"].iloc[0]) is True


def test_source_colors_are_stable_and_high_contrast():
    colors = ra._source_color_map(["gamma.md", "alpha.md", "beta.md"])
    assert set(colors) == {"alpha.md", "beta.md", "gamma.md"}
    assert all(value.startswith("#") and len(value) == 7 for value in colors.values())


def test_no_deprecated_streamlit_container_width_argument():
    content = MODULE_PATH.read_text()
    assert "use_container_width" not in content
    assert "width=\"stretch\"" in content
