#!/usr/bin/env python3
"""Phase 8 deployment readiness checks.

This script is intentionally safe: it reads config and filesystem state only.
It does not create AWS resources or mutate project data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.runtime import get_runtime_settings


ARTIFACTS = {
    "gold_claim_features": "data/gold/gold_claim_features.parquet",
    "inference_artifacts": "data/gold/inference_artifacts.json",
    "training_report": "models/training_report.json",
    "xgb_model": "models/xgb_model.pkl",
    "vector_metadata": "data/vector_store/policy_metadata.json",
    "policy_chunks": "data/policies/processed/policy_chunks.parquet",
}

REQUIRED_FILES = [
    "Dockerfile.api",
    "Dockerfile.streamlit",
    "docker-compose.yml",
    ".env.docker.example",
    ".env.aws.example",
    "requirements.txt",
    "requirements-api.txt",
]


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return result.returncode == 0, result.stdout.strip()
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    settings = get_runtime_settings()
    print("Phase 8 deployment readiness")
    print(f"  APP_ENV:            {settings.app_env}")
    print(f"  Root:               {settings.root_dir}")
    print(f"  Auth backend:       {settings.auth_backend}")
    print(f"  UI API base URL:    {settings.ui_api_base_url}")
    print(f"  OpenAI enabled:     {settings.enable_openai}")
    print(f"  OpenAI key present: {settings.openai_api_key_configured}")
    print(f"  S3 bucket:          {settings.s3_bucket_name or 'not configured'}")

    print("\nRequired deployment files:")
    missing_files = []
    for rel in REQUIRED_FILES:
        exists = (ROOT / rel).exists()
        print(f"  {'OK     ' if exists else 'MISSING'} {rel}")
        if not exists:
            missing_files.append(rel)

    print("\nRuntime artifacts:")
    missing_artifacts = []
    for name, rel in ARTIFACTS.items():
        exists = (ROOT / rel).exists()
        print(f"  {'OK     ' if exists else 'MISSING'} {name:<22} {rel}")
        if not exists:
            missing_artifacts.append(name)

    print("\nDocker availability:")
    docker_ok = shutil.which("docker") is not None
    print(f"  {'OK     ' if docker_ok else 'MISSING'} docker executable")
    compose_ok = False
    if docker_ok:
        compose_ok, compose_out = _run(["docker", "compose", "version"])
        print(f"  {'OK     ' if compose_ok else 'MISSING'} docker compose plugin")
        if compose_out:
            print(f"         {compose_out.splitlines()[0]}")

    if settings.app_env == "aws" and settings.auth_backend != "postgresql":
        print("\nERROR: APP_ENV=aws requires AUTH_DATABASE_URL to use PostgreSQL/RDS.")
        return 2

    report = {
        "app_env": settings.app_env,
        "auth_backend": settings.auth_backend,
        "missing_deployment_files": missing_files,
        "missing_artifacts": missing_artifacts,
        "docker_available": docker_ok,
        "docker_compose_available": compose_ok,
    }
    out_path = ROOT / "logs" / "phase8_deployment_readiness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReadiness report written to {out_path}")

    if missing_files:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
