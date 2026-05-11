from pathlib import Path

import numpy as np

from src.rag.embedder import TfidfTextEmbedder, create_embedder, embedder_from_vector_metadata
from src.rag.vector_store import LocalVectorStore


def test_tfidf_embedder_fits_persists_and_reloads(tmp_path: Path):
    texts = [
        "diagnosis code is required to support medical necessity",
        "prior authorization is required for selected services",
        "high cost claims require supporting documentation",
    ]
    embedder = TfidfTextEmbedder(artifact_dir=tmp_path, max_features=64)
    vectors = embedder.embed_texts(texts)

    assert vectors.shape[0] == 3
    assert vectors.shape[1] <= 64
    assert (tmp_path / "policy_tfidf_vectorizer.pkl").exists()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    reloaded = TfidfTextEmbedder(artifact_dir=tmp_path, max_features=64)
    q = reloaded.embed_query("diagnosis medical necessity")
    assert q.shape[0] == vectors.shape[1]


def test_tfidf_roundtrip_with_vector_metadata(tmp_path: Path):
    texts = [
        "diagnosis code required for claim adjudication",
        "billing amount above benchmark requires documentation",
    ]
    metadata = [
        {"chunk_id": "c1", "chunk_text": texts[0], "policy_tags_json": '["diagnosis"]'},
        {"chunk_id": "c2", "chunk_text": texts[1], "policy_tags_json": '["billing"]'},
    ]
    embedder = create_embedder(backend="tfidf", artifact_dir=tmp_path, tfidf_max_features=128)
    vectors = embedder.embed_texts(texts)
    store = LocalVectorStore(vector_dir=tmp_path, vector_backend="numpy").build(
        embeddings=vectors,
        metadata=metadata,
        embedding_backend=embedder.metadata()["embedding_backend"],
        embedding_model=embedder.metadata()["embedding_model"],
        embedding_metadata=embedder.metadata(),
    )
    assert store["embedding_backend"] == "tfidf"

    query_embedder = embedder_from_vector_metadata(vector_dir=tmp_path)
    query_vector = query_embedder.embed_query("diagnosis support for claim")
    loaded_store = LocalVectorStore(vector_dir=tmp_path, vector_backend="numpy").load()
    results = loaded_store.search(query_vector, top_k=1)
    assert results[0].chunk_id == "c1"
