"""Health and artifact status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_current_user, settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    return {"status": "ok", "service": "claim-denial-api"}


@router.get("/artifacts")
def artifact_health(user: dict = Depends(get_current_user)) -> dict:
    s = settings()
    checks = {
        "gold_claim_features": (s.gold_dir / "gold_claim_features.parquet").exists(),
        "inference_artifacts": (s.gold_dir / "inference_artifacts.json").exists(),
        "training_report": (s.models_dir / "training_report.json").exists(),
        "xgb_model": (s.models_dir / "xgb_model.pkl").exists(),
        "vector_metadata": (s.vector_dir / "policy_metadata.json").exists(),
    }
    return {"status": "success", "role": user.get("role"), "artifacts": checks}
