import pandas as pd

from dev_dashboard.tabs.policy_rag import _deduplicate_policy_matches, _format_percent


def test_deduplicate_policy_matches_keeps_best_score():
    df = pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "reason_code": "MISSING_DIAGNOSIS",
                "policy_chunk_id": "P1",
                "similarity_score": 0.55,
                "policy_text": "lower score duplicate",
            },
            {
                "claim_id": "C1",
                "reason_code": "MISSING_DIAGNOSIS",
                "policy_chunk_id": "P1",
                "similarity_score": 0.91,
                "policy_text": "best score duplicate",
            },
        ]
    )
    out = _deduplicate_policy_matches(df)
    assert len(out) == 1
    assert out.iloc[0]["policy_text"] == "best score duplicate"


def test_format_percent_accepts_probability_or_percent():
    assert _format_percent(0.3076) == "30.76%"
    assert _format_percent(30.76) == "30.76%"
