import pandas as pd

from src.rag.policy_matcher import PolicyMatcher
from src.rag.schemas import POLICY_MATCH_TABLE, FINAL_EXPLANATION_TABLE


class FakeStore:
    def __init__(self, explanation_df):
        self.explanation_df = explanation_df
        self.writes = {}

    def read_table(self, name):
        return self.explanation_df

    def write_table(self, name, df):
        self.writes[name] = df
        return f"memory://{name}"


class FakeRetriever:
    def retrieve(self, *, query, policy_tags=None, top_k=3, min_score=0.25):
        from src.rag.schemas import PolicySearchResult
        return [
            PolicySearchResult(
                chunk_id="chunk1",
                score=0.91,
                raw_score=0.87,
                tag_overlap_count=1,
                metadata={
                    "chunk_id": "chunk1",
                    "chunk_text": "Diagnosis is required to support medical necessity.",
                    "source_name": "policy.md",
                    "source_type": "md",
                    "source_path": "policy.md",
                    "section_title": "Diagnosis Policy",
                    "page_number": None,
                    "policy_tags_json": '["diagnosis"]',
                },
            )
        ]


def test_policy_matcher_writes_match_and_final_outputs(tmp_path):
    explanation_df = pd.DataFrame([
        {
            "claim_id": "C1",
            "risk_score": 0.8,
            "risk_level": "HIGH",
            "predicted_denial": 1,
            "reason_rank": 1,
            "reason_code": "MISSING_DIAGNOSIS",
            "reason_title": "Missing diagnosis code",
            "reason_text": "The claim is missing diagnosis support.",
            "fix_suggestion": "Add a valid diagnosis code.",
            "policy_query": "Diagnosis code required for medical necessity.",
            "policy_tags": '["diagnosis"]',
        }
    ])
    store = FakeStore(explanation_df)
    matcher = PolicyMatcher(
        gold_dir=tmp_path,
        vector_dir=tmp_path,
        table_store=store,
        retriever=FakeRetriever(),
        top_k=1,
    )
    report = matcher.run()
    assert report["status"] == "success"
    assert POLICY_MATCH_TABLE in store.writes
    assert FINAL_EXPLANATION_TABLE in store.writes
    assert not store.writes[POLICY_MATCH_TABLE].empty
    assert not store.writes[FINAL_EXPLANATION_TABLE].empty
