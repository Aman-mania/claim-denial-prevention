from src.rag.embedder import OpenAITextEmbedder, create_embedder


class _Item:
    def __init__(self, embedding):
        self.embedding = embedding


class _Response:
    def __init__(self, rows):
        self.data = [_Item(row) for row in rows]


class _Embeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        rows = []
        for i, _ in enumerate(kwargs["input"]):
            rows.append([1.0 + i, 0.0, 0.0])
        return _Response(rows)


class _Client:
    def __init__(self):
        self.embeddings = _Embeddings()


def test_openai_embedder_uses_client_and_normalizes_rows():
    client = _Client()
    embedder = OpenAITextEmbedder(client=client, model_name="text-embedding-3-small", normalize_embeddings=True)
    vectors = embedder.embed_texts(["policy one", "policy two"])

    assert vectors.shape == (2, 3)
    assert round(float(vectors[0][0]), 4) == 1.0
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert client.embeddings.calls[0]["input"] == ["policy one", "policy two"]
    assert embedder.metadata()["embedding_backend"] == "openai"


def test_create_embedder_supports_openai_backend():
    embedder = create_embedder(backend="openai", model_name="text-embedding-3-small", allow_fallback=False)
    assert isinstance(embedder, OpenAITextEmbedder)
