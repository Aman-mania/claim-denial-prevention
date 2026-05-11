from src.rag.chunker import PolicyChunker, infer_policy_tags
from src.rag.schemas import PolicyDocument


def test_infer_policy_tags_detects_reason_tags():
    tags = infer_policy_tags("A diagnosis code is required for medical necessity documentation.")
    assert "diagnosis" in tags
    assert "medical_necessity" in tags
    assert "documentation" in tags


def test_policy_chunker_creates_chunks_with_metadata():
    doc = PolicyDocument(
        document_id="doc1",
        source_name="policy.md",
        source_type="md",
        source_path="policy.md",
        text="# Diagnosis Policy\nDiagnosis code is required for medical necessity. " * 20,
    )
    chunker = PolicyChunker(chunk_size_words=40, chunk_overlap_words=5)
    chunks = chunker.chunk_documents([doc])
    assert chunks
    assert chunks[0].source_name == "policy.md"
    assert "diagnosis" in chunks[0].policy_tags
    df = chunker.to_dataframe(chunks)
    assert "chunk_text" in df.columns
    assert "policy_tags_json" in df.columns
