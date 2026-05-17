from pathlib import Path


def test_docker_env_defaults_to_openai_and_sklearn():
    content = Path(".env.docker.example").read_text()
    assert "RAG_EMBEDDING_BACKEND=openai" in content
    assert "RAG_VECTOR_BACKEND=sklearn" in content
    assert "OPENAI_API_KEY=" in content


def test_docker_api_requirements_include_openai_not_sentence_transformers():
    content = Path("requirements-docker-api.txt").read_text()
    assert "openai" in content
    assert "sentence-transformers" not in content
    assert "faiss-cpu" not in content
