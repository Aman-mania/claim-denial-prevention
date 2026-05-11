from pathlib import Path

from src.rag.document_loader import PolicyDocumentLoader


def test_document_loader_reads_markdown(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "policy.md").write_text("# Policy\nDiagnosis is required.", encoding="utf-8")
    docs = PolicyDocumentLoader().load_documents(raw)
    assert len(docs) == 1
    assert docs[0].source_type == "md"
    assert "Diagnosis" in docs[0].text
