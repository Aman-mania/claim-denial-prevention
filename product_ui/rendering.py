"""Rendering helpers for role-aware Streamlit UI.

Helpers are kept pure where possible so they can be unit-tested without running
Streamlit.
"""

from __future__ import annotations

from typing import Any


def risk_badge_text(prediction: dict[str, Any] | None) -> str:
    if not prediction:
        return "Risk unavailable"
    level = str(prediction.get("risk_level") or "UNKNOWN").upper()
    score = prediction.get("risk_score")
    try:
        score_txt = f"{float(score):.1%}"
    except (TypeError, ValueError):
        score_txt = "n/a"
    return f"{level} · {score_txt}"


def visible_tabs_for_role(role: str | None) -> list[str]:
    normalized = str(role or "").lower()
    if normalized == "developer":
        return [
            "Overview",
            "Custom Claim",
            "Data Pipeline",
            "Risk Model",
            "Risk Explanations",
            "Policy Evidence",
            "Retrieval Analytics",
            "System Health",
        ]
    return ["Overview", "Claim Analytics", "Custom Claim", "System Health"]


def analyst_result_sections(result: dict[str, Any]) -> dict[str, int]:
    data = result.get("data") if result.get("data") else result
    return {
        "warnings": len(((data.get("validation") or {}).get("warnings") or [])),
        "reasons": len(data.get("reasons") or []),
        "policy_evidence": len(data.get("policy_evidence") or []),
        "recommendations": len(data.get("recommendations") or []),
        "action_plan": len(((data.get("agent_presentation") or {}).get("action_plan") or [])),
    }


def short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def dedupe_policy_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest policy evidence per source/section/reason tuple."""
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in evidence:
        key = (
            str(item.get("reason_code") or ""),
            str(item.get("source_name") or ""),
            str(item.get("section_title") or item.get("policy_chunk_id") or ""),
        )
        current = best.get(key)
        if current is None:
            best[key] = item
            continue
        try:
            new_score = float(item.get("similarity_score") or 0)
            old_score = float(current.get("similarity_score") or 0)
        except (TypeError, ValueError):
            new_score = old_score = 0
        if new_score > old_score:
            best[key] = item
    return sorted(best.values(), key=lambda row: float(row.get("similarity_score") or 0), reverse=True)


def artifact_health_counts(payload: dict[str, Any] | None) -> dict[str, int]:
    """Return simple readiness counts from the protected artifact-health API payload."""
    artifacts = ((payload or {}).get("artifacts") or {})
    total = len(artifacts)
    ready = sum(1 for value in artifacts.values() if bool(value))
    missing = max(total - ready, 0)
    return {"total": total, "ready": ready, "missing": missing}


def overall_health_label(api_ok: bool, artifact_payload: dict[str, Any] | None) -> str:
    """Business-friendly health label for the product UI."""
    if not api_ok:
        return "Unavailable"
    counts = artifact_health_counts(artifact_payload)
    if counts["total"] and counts["missing"]:
        return "Degraded"
    return "Ready"


def artifact_rows_from_payload(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert artifact-health JSON into table rows without exposing raw JSON to analysts."""
    artifacts = ((payload or {}).get("artifacts") or {})
    rows: list[dict[str, Any]] = []
    for name, exists in artifacts.items():
        rows.append({
            "artifact": str(name).replace("_", " ").title(),
            "status": "Available" if bool(exists) else "Missing",
            "ready": bool(exists),
        })
    return rows


def deployment_readiness_rows(*, api_ok: bool, artifact_payload: dict[str, Any] | None, openai_enabled: bool = False) -> list[dict[str, str]]:
    """Return a compact readiness checklist for developer UI."""
    counts = artifact_health_counts(artifact_payload)
    artifact_ready = counts["total"] > 0 and counts["missing"] == 0
    return [
        {"check": "FastAPI backend", "status": "Ready" if api_ok else "Unavailable", "environment": "local / EC2"},
        {"check": "Authentication", "status": "Ready", "environment": "SQLite locally, RDS on AWS"},
        {"check": "Model + RAG artifacts", "status": "Ready" if artifact_ready else "Needs attention", "environment": "local files now, S3 backup on AWS"},
        {"check": "OpenAI presentation", "status": "Enabled" if openai_enabled else "Optional / disabled", "environment": "Secrets Manager on AWS"},
        {"check": "RDS PostgreSQL", "status": "Pending AWS setup", "environment": "AWS only"},
        {"check": "S3 artifact bucket", "status": "Pending AWS setup", "environment": "AWS only"},
    ]
