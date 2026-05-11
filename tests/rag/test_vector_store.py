import numpy as np
import pytest

from src.rag.vector_store import LocalFaissVectorStore, LocalVectorStore


@pytest.mark.week6
def test_numpy_vector_store_round_trip_without_faiss(tmp_path):
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    metadata = [
        {"chunk_id": "a", "chunk_text": "diagnosis policy", "policy_tags_json": '["diagnosis"]'},
        {"chunk_id": "b", "chunk_text": "cost policy", "policy_tags_json": '["high_cost"]'},
    ]
    store = LocalVectorStore(vector_dir=tmp_path, vector_backend="numpy")
    info = store.build(embeddings=embeddings, metadata=metadata)

    assert info["vector_backend"] == "numpy"
    assert (tmp_path / "policy_vectors.npy").exists()
    assert (tmp_path / "policy_metadata.json").exists()

    loaded = LocalVectorStore(vector_dir=tmp_path, vector_backend="numpy").load()
    results = loaded.search(np.asarray([1.0, 0.0], dtype="float32"), top_k=1)
    assert results[0].chunk_id == "a"


@pytest.mark.week6
@pytest.mark.slow
def test_local_faiss_vector_store_round_trip_when_faiss_available(tmp_path):
    pytest.importorskip("faiss")
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    metadata = [
        {"chunk_id": "a", "chunk_text": "diagnosis policy", "policy_tags_json": '["diagnosis"]'},
        {"chunk_id": "b", "chunk_text": "cost policy", "policy_tags_json": '["high_cost"]'},
    ]
    store = LocalFaissVectorStore(vector_dir=tmp_path, vector_backend="faiss")
    info = store.build(embeddings=embeddings, metadata=metadata)

    assert info["vector_backend"] == "faiss"
    assert (tmp_path / "policy.faiss").exists()
    assert (tmp_path / "policy_vectors.npy").exists()

    loaded = LocalFaissVectorStore(vector_dir=tmp_path).load()
    results = loaded.search(np.asarray([1.0, 0.0], dtype="float32"), top_k=1)
    assert results[0].chunk_id == "a"
