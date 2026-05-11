import pandas as pd

from src.io.table_contracts import enforce_table_contract


def test_policy_chunks_contract_uses_cost_safe_dtypes():
    df = pd.DataFrame([
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "source_name": "p.md",
            "source_type": "md",
            "source_path": "p.md",
            "section_title": "Policy",
            "page_number": None,
            "chunk_index": 0,
            "chunk_text": "Diagnosis required.",
            "policy_tags_json": '["diagnosis"]',
            "token_estimate": 10,
            "embedding_model": "m",
            "rag_version": "week6_rag_v1",
            "created_at": "now",
        }
    ])
    out = enforce_table_contract(df, "policy_chunks")
    assert str(out["chunk_text"].dtype) == "string"
    assert str(out["chunk_index"].dtype) == "int32"
    assert str(out["page_number"].dtype) == "Int32"
