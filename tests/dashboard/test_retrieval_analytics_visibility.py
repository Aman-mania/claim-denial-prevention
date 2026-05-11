from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


class _FakeStreamlit(types.SimpleNamespace):
    def cache_data(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


# The tests exercise pure helper functions; a light Streamlit stub keeps them
# runnable in minimal CI environments where Streamlit is not installed.
sys.modules.setdefault("streamlit", _FakeStreamlit())

MODULE_PATH = Path(__file__).resolve().parents[2] / "dev_dashboard" / "tabs" / "retrieval_analytics.py"
if not MODULE_PATH.exists():
    # When tests are copied into the repo, parents[2] points to repo root.
    MODULE_PATH = Path.cwd() / "dev_dashboard" / "tabs" / "retrieval_analytics.py"

spec = importlib.util.spec_from_file_location("retrieval_analytics", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_source_color_map_is_high_contrast_and_stable() -> None:
    colors = module._source_color_map([
        "hipaa_operational_safeguards_pack.md",
        "official_policy_source_registry_summary.md",
        "payer_policy_reference_pack.md",
        "sample_claim_denial_policy_pack.md",
        "us_healthcare_claim_policy_seed_pack.md",
    ])
    assert len(colors) == 5
    assert all(color.startswith("#") for color in colors.values())
    # Ensure no white/near-white colors are assigned, which was the UI issue.
    assert "#FFFFFF" not in {c.upper() for c in colors.values()}
    assert "#F8FAFC" not in {c.upper() for c in colors.values()}


def test_source_utilization_includes_sources_with_zero_retrievals() -> None:
    chunks = pd.DataFrame(
        {
            "chunk_id": ["a1", "a2", "b1", "c1"],
            "source_name": ["A.md", "A.md", "B.md", "C.md"],
            "section_title": ["A1", "A2", "B1", "C1"],
            "chunk_text": ["a", "a", "b", "c"],
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "reason_code": ["R1", "R2"],
            "policy_chunk_id": ["a1", "a2"],
            "source_name": ["A.md", "A.md"],
            "similarity_score": [0.8, 0.7],
        }
    )

    util = module._prepare_source_utilization(chunks, matches)
    assert set(util["source_name"]) == {"A.md", "B.md", "C.md"}
    zero_sources = set(util.loc[util["retrieval_count"] == 0, "source_name"])
    assert zero_sources == {"B.md", "C.md"}


def test_vector_projection_keeps_all_policy_sources_visible() -> None:
    vectors = np.eye(5, dtype=np.float32)
    chunks = pd.DataFrame(
        {
            "chunk_id": [f"ch{i}" for i in range(5)],
            "source_name": ["A.md", "B.md", "C.md", "D.md", "E.md"],
            "section_title": [f"S{i}" for i in range(5)],
            "chunk_text": ["text"] * 5,
        }
    )
    matches = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "reason_code": ["R1"],
            "policy_chunk_id": ["ch2"],
            "source_name": ["C.md"],
            "similarity_score": [0.9],
        }
    )

    projection = module._prepare_vector_projection(vectors, chunks, matches, selected_claim="C1", method="PCA")
    assert set(projection["source_name"]) == {"A.md", "B.md", "C.md", "D.md", "E.md"}
    assert projection["selected_claim_match"].sum() == 1
