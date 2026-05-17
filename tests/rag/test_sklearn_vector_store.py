import numpy as np

from src.rag.vector_store import LocalVectorStore


def test_sklearn_vector_store_build_load_search(tmp_path):
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    metadata = [
        {"chunk_id": "a", "chunk_text": "alpha"},
        {"chunk_id": "b", "chunk_text": "beta"},
        {"chunk_id": "c", "chunk_text": "mixed"},
    ]

    store = LocalVectorStore(vector_dir=tmp_path, vector_backend="sklearn")
    info = store.build(
        embeddings=vectors,
        metadata=metadata,
        embedding_backend="openai",
        embedding_model="text-embedding-3-small",
    )

    assert info["vector_backend"] == "sklearn"
    assert info["sklearn_index_written"] is True
    assert (tmp_path / "policy_sklearn_nn.pkl").exists()

    loaded = LocalVectorStore(vector_dir=tmp_path, vector_backend="sklearn").load()
    results = loaded.search(np.asarray([1.0, 0.0], dtype="float32"), top_k=2)

    assert [r.chunk_id for r in results][:1] == ["a"]
    assert results[0].score > results[1].score
