import json

import numpy as np
import pytest

from src.rag.embedder import HashingTextEmbedder, embedder_from_vector_metadata
from src.rag.vector_store import LocalVectorStore


@pytest.mark.week6
def test_retriever_reuses_hashing_backend_from_vector_metadata(tmp_path):
    embedder = HashingTextEmbedder(n_features=128)
    texts = ["diagnosis medical necessity policy", "high cost documentation policy"]
    embeddings = embedder.embed_texts(texts)
    metadata = [
        {"chunk_id": "diag", "chunk_text": texts[0], "policy_tags_json": '["diagnosis"]'},
        {"chunk_id": "cost", "chunk_text": texts[1], "policy_tags_json": '["high_cost"]'},
    ]
    LocalVectorStore(vector_dir=tmp_path, vector_backend="numpy").build(
        embeddings=embeddings,
        metadata=metadata,
        embedding_backend=embedder.metadata()["embedding_backend"],
        embedding_model=embedder.metadata()["embedding_model"],
        embedding_metadata=embedder.metadata(),
    )

    query_embedder = embedder_from_vector_metadata(vector_dir=tmp_path)
    assert isinstance(query_embedder, HashingTextEmbedder)
    assert query_embedder.n_features == 128


@pytest.mark.week6
def test_sentence_transformer_index_does_not_silently_fallback_to_hashing(tmp_path, monkeypatch):
    payload = {
        "embedding_backend": "sentence-transformers",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "vector_backend": "numpy",
        "metadata": [],
    }
    (tmp_path / "policy_metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embedder = embedder_from_vector_metadata(vector_dir=tmp_path)
    with pytest.raises(Exception):
        embedder.embed_query("diagnosis policy")
