from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase8_deployment_files_exist():
    for rel in [
        "Dockerfile.api",
        "Dockerfile.streamlit",
        "docker-compose.yml",
        ".env.docker.example",
        ".env.aws.example",
        "scripts/check_phase8_deployment.py",
        "scripts/docker_smoke_test.sh",
        "docs/PHASE8_LOCAL_AND_AWS_READY_DEPLOYMENT.md",
    ]:
        assert (ROOT / rel).exists(), rel


def test_docker_compose_declares_api_and_ui_services():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "api:" in text
    assert "product-ui:" in text
    assert "CLAIM_DENIAL_API_BASE_URL: http://api:8000" in text
