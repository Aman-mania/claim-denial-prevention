import numpy as np

from src.rag.embedder import HashingTextEmbedder, create_embedder


def test_hashing_embedder_is_deterministic_and_l2_normalized():
    embedder = HashingTextEmbedder(n_features=128)
    texts = ["diagnosis required for medical necessity", "prior authorization required"]

    a = embedder.embed_texts(texts)
    b = embedder.embed_texts(texts)

    assert a.shape == (2, 128)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), np.ones(2))
    assert embedder.metadata()["embedding_backend"] == "sklearn-hashing"


def test_create_embedder_supports_explicit_hashing_backend():
    embedder = create_embedder(backend="sklearn-hashing", hashing_features=64)
    vector = embedder.embed_query("missing diagnosis policy")

    assert vector.shape == (64,)
    assert embedder.metadata()["embedding_model"] == "sklearn-hashing-64"
