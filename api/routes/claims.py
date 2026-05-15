"""Claim validation and recommendation routes."""

from __future__ import annotations

import copy
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_auth_repository, get_current_user, get_remediation_agent, settings
from api.schemas import ClaimRequest
from src.auth.repository import AuthRepository
from src.rules.claim_validator import ClaimInputValidator

router = APIRouter(prefix="/claims", tags=["claims"])


_ANALYST_REASON_FIELDS = {
    "reason_code",
    "reason_rank",
    "reason_title",
    "reason_text",
    "fix_suggestion",
}

_ANALYST_POLICY_FIELDS = {
    "reason_code",
    "policy_chunk_id",
    "source_name",
    "source_type",
    "section_title",
    "page_number",
    "policy_summary",
    "similarity_score",
}

_ANALYST_PREDICTION_FIELDS = {
    "claim_id",
    "risk_score",
    "risk_level",
    "predicted_denial",
    "classification_threshold",
    "review_threshold",
    "model_used",
}


def _claim_dict(payload: ClaimRequest) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _pick_fields(row: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in allowed if key in row}


def _redact_for_role(result: dict[str, Any], role: str | None) -> dict[str, Any]:
    """Hide technical/debug fields from business analyst responses.

    The developer role keeps the full response for debugging. The analyst role
    receives the business-facing subset used by the role-aware Streamlit UI.
    """
    if str(role or "").lower() == "developer":
        return result

    out = copy.deepcopy(result)
    out.pop("features", None)

    prediction = out.get("prediction")
    if isinstance(prediction, dict):
        out["prediction"] = _pick_fields(prediction, _ANALYST_PREDICTION_FIELDS)

    reasons = out.get("reasons") or []
    if isinstance(reasons, list):
        out["reasons"] = [
            _pick_fields(reason, _ANALYST_REASON_FIELDS)
            for reason in reasons
            if isinstance(reason, dict)
        ]

    evidence = out.get("policy_evidence") or []
    if isinstance(evidence, list):
        out["policy_evidence"] = [
            _pick_fields(item, _ANALYST_POLICY_FIELDS)
            for item in evidence
            if isinstance(item, dict)
        ]

    out["response_scope"] = "business_analyst"
    return out


@router.post("/validate")
def validate_claim(
    payload: ClaimRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    validator = ClaimInputValidator.from_gold_dir(settings().gold_dir)
    result = validator.validate(_claim_dict(payload))
    return {"status": "success", "user_role": user.get("role"), "validation": result.to_dict()}


@router.post("/recommend")
def recommend_claim(
    payload: ClaimRequest,
    user: dict = Depends(get_current_user),
    repo: AuthRepository = Depends(get_auth_repository),
) -> dict:
    claim = _claim_dict(payload)
    agent = get_remediation_agent()
    result = agent.analyze_claim(claim)
    repo.record_audit(
        user=user,
        action="claims.recommend",
        claim_id=str(claim.get("claim_id")),
        status=str(result.get("status")),
        metadata=json.dumps({"risk_level": (result.get("prediction") or {}).get("risk_level")}, default=str),
    )
    if result.get("status") == "blocked":
        return {"status": "blocked", "data": _redact_for_role(result, user.get("role"))}
    if result.get("status") == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result)
    return {"status": "success", "data": _redact_for_role(result, user.get("role"))}
