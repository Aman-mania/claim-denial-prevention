from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_lightweight_docker_requirements_exist_and_exclude_semantic_heavy_packages():
    api = (ROOT / "requirements-docker-api.txt").read_text(encoding="utf-8")
    ui = (ROOT / "requirements-docker-ui.txt").read_text(encoding="utf-8")
    assert "fastapi" in api
    assert "streamlit" in ui
    assert "sentence-transformers" not in api
    assert "sentence-transformers" not in ui
    assert "faiss-cpu" not in api
    assert "faiss-cpu" not in ui


def test_dockerfiles_use_lightweight_requirements():
    api = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    ui = (ROOT / "Dockerfile.streamlit").read_text(encoding="utf-8")
    assert "requirements-docker-api.txt" in api
    assert "requirements-docker-ui.txt" in ui
    assert "INSTALL_SEMANTIC" in api
