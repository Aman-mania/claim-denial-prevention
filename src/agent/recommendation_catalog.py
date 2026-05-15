"""Deterministic remediation recommendations for claim-denial prevention.

This catalog is the source of truth for actions. The optional OpenAI layer may
rewrite these actions into a readable presentation, but it must not invent or
replace them.
"""

from __future__ import annotations

from typing import Any

from src.agent.schemas import AgentDecision, Recommendation

_REASON_RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "MISSING_DIAGNOSIS": (
        "Add or verify the ICD diagnosis code before submission.",
        "The claim lacks diagnosis support needed to justify the billed service.",
    ),
    "MISSING_PROCEDURE": (
        "Add the billed procedure code before submission.",
        "The claim is incomplete without a procedure/service code.",
    ),
    "PROCEDURE_WITHOUT_DIAGNOSIS": (
        "Link the procedure to a supporting diagnosis code.",
        "A billed procedure generally needs medical-necessity support.",
    ),
    "DIAGNOSIS_WITHOUT_PROCEDURE": (
        "Add the performed procedure or remove unsupported diagnosis-only claim data.",
        "A diagnosis-only claim line is incomplete for billing review.",
    ),
    "HIGH_BILLING_AMOUNT": (
        "Review the billed amount against the expected benchmark and attach justification if appropriate.",
        "The billed amount is materially above expected cost benchmarks.",
    ),
    "HIGH_COST_CLAIM": (
        "Check prior authorization and documentation requirements for the high-cost claim.",
        "High-cost claims often require additional documentation or authorization.",
    ),
    "WEAK_COST_BENCHMARK": (
        "Verify procedure, provider location, and cost benchmark before submission.",
        "The available cost benchmark is not a strong match for the claim context.",
    ),
    "PROVIDER_VOLUME_RISK": (
        "Route this claim for manual review because provider activity volume is elevated.",
        "High-volume provider patterns can indicate duplicate, batch, or incomplete submissions.",
    ),
    "PROVIDER_HISTORY_RISK": (
        "Use the claim-quality checklist before submission because provider history indicates elevated risk.",
        "Provider history has contributed to higher claim-quality risk.",
    ),
    "PATIENT_FREQUENCY_RISK": (
        "Review for duplicate, repeated, or unusually frequent claims for this patient.",
        "High patient claim frequency can increase payer review risk.",
    ),
    "DIAGNOSIS_SEVERITY_SUPPORT": (
        "Ensure clinical documentation supports the diagnosis severity.",
        "Medium/high severity diagnoses may need stronger supporting records.",
    ),
    "SPECIALTY_CREDENTIAL_CHECK": (
        "Verify that the provider specialty and credentials align with the billed service.",
        "Specialty mismatch can increase denial or manual-review risk.",
    ),
}

_WARNING_RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "CLAIM_WARN_MISSING_DIAGNOSIS": (
        "Add a diagnosis code if available before submitting the claim.",
        "Missing diagnosis is a non-blocking warning but can increase denial risk.",
    ),
    "CLAIM_WARN_MISSING_PROCEDURE": (
        "Add a procedure code if the billed service is known.",
        "Missing procedure is a non-blocking warning but makes the claim incomplete.",
    ),
    "CLAIM_WARN_AMOUNT_MISSING": (
        "Verify and enter billed amount when available.",
        "The model can impute amount, but real amount improves review quality.",
    ),
    "CLAIM_WARN_UNKNOWN_PROVIDER": (
        "Confirm the provider ID and add/update provider reference data if needed.",
        "Unknown provider IDs use safe defaults and reduce provider-history confidence.",
    ),
    "CLAIM_WARN_UNKNOWN_DIAGNOSIS": (
        "Confirm the diagnosis code and update diagnosis reference data if needed.",
        "Unknown diagnosis codes use safe defaults and reduce severity-confidence.",
    ),
    "CLAIM_WARN_UNKNOWN_PROCEDURE": (
        "Confirm the procedure code and update cost benchmark data if needed.",
        "Unknown procedure codes reduce cost-benchmark confidence.",
    ),
}


def _priority_for_risk(risk_level: str | None) -> str:
    level = str(risk_level or "LOW").upper()
    if level == "HIGH":
        return "HIGH"
    if level == "MEDIUM":
        return "MEDIUM"
    return "LOW"


class RecommendationCatalog:
    """Build deterministic recommendations from validation + reasons + policy."""

    def generate(
        self,
        *,
        validation: dict[str, Any],
        prediction: dict[str, Any] | None,
        reasons: list[dict[str, Any]],
        policy_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        risk_priority = _priority_for_risk((prediction or {}).get("risk_level"))
        recommendations: list[Recommendation] = []
        seen_actions: set[str] = set()

        def add(action: str, priority: str, why: str, reason_code: str | None = None, evidence_source: str | None = None) -> None:
            if action in seen_actions:
                return
            seen_actions.add(action)
            recommendations.append(Recommendation(
                action=action,
                priority=priority if priority in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM",  # type: ignore[arg-type]
                reason_code=reason_code,
                why=why,
                evidence_source=evidence_source,
            ))

        for issue in validation.get("blocking_errors", []) or []:
            field = issue.get("field") or "claim input"
            add(
                action=f"Fix required input field before scoring: {field}.",
                priority="HIGH",
                why=issue.get("message") or "The claim cannot be safely processed until this input issue is fixed.",
                reason_code=issue.get("code"),
            )

        for reason in reasons:
            code = str(reason.get("reason_code") or "")
            action, why = _REASON_RECOMMENDATIONS.get(
                code,
                (reason.get("fix_suggestion") or "Review this claim before submission.", reason.get("reason_text") or "The model identified this as a relevant risk factor."),
            )
            source = None
            for evidence in policy_evidence:
                if evidence.get("reason_code") == code:
                    source = evidence.get("source_name")
                    break
            priority = "HIGH" if risk_priority == "HIGH" else ("MEDIUM" if risk_priority == "MEDIUM" else "LOW")
            add(action=action, priority=priority, why=why, reason_code=code or None, evidence_source=source)

        for warning in validation.get("warnings", []) or []:
            code = str(warning.get("code") or "")
            action_why = _WARNING_RECOMMENDATIONS.get(code)
            if not action_why:
                continue
            action, why = action_why
            add(action=action, priority="MEDIUM" if risk_priority in {"HIGH", "MEDIUM"} else "LOW", why=why, reason_code=code)

        if not recommendations and prediction:
            risk_level = str(prediction.get("risk_level", "LOW")).upper()
            if risk_level == "LOW":
                add(
                    action="Proceed with normal submission after standard claim checks.",
                    priority="LOW",
                    why="The model classified the claim as low risk and no blocking validation errors were found.",
                )
            else:
                add(
                    action="Route this claim for analyst review before submission.",
                    priority=risk_priority,
                    why="The model classified the claim as needing additional review.",
                )

        return [item.to_dict() for item in recommendations]

    def decision(self, *, validation: dict[str, Any], prediction: dict[str, Any] | None, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        if validation.get("blocking_errors"):
            return AgentDecision(
                status="BLOCKED",
                priority="HIGH",
                summary="Claim cannot be scored/submitted until blocking input errors are corrected.",
            ).to_dict()

        risk_level = str((prediction or {}).get("risk_level", "LOW")).upper()
        if risk_level == "HIGH":
            return AgentDecision(
                status="REVIEW_REQUIRED",
                priority="HIGH",
                summary="High denial risk. Analyst review and remediation are required before submission.",
            ).to_dict()
        if risk_level == "MEDIUM":
            return AgentDecision(
                status="REVIEW_RECOMMENDED",
                priority="MEDIUM",
                summary="Moderate denial risk. Review recommended before submission.",
            ).to_dict()
        if any(item.get("priority") in {"HIGH", "MEDIUM"} for item in recommendations):
            return AgentDecision(
                status="REVIEW_RECOMMENDED",
                priority="MEDIUM",
                summary="Low model risk, but validation/recommendation warnings deserve review.",
            ).to_dict()
        return AgentDecision(
            status="READY_TO_SUBMIT",
            priority="LOW",
            summary="No blocking issues were found and model risk is low.",
        ).to_dict()
